from abc import ABC, abstractmethod
import torch

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv

import numpy as np

class MultiQAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, model, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001,
                 recurrence=4,
                 adam_eps=1e-8,
                 preprocess_obss=None, reshape_reward=None):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        num_frames_per_proc = num_frames_per_proc or 8  # is 8 correct here?

        self.env = ParallelEnv(envs)
        self.model = model
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        self.reward_size = self.model.reward_size

        # Control parameters

        assert self.model.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.model.to(self.device)
        self.model.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        if self.model.recurrent:
            self.memory = torch.zeros(shape[1], self.model.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.model.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
#        self.masks = torch.zeros(*shape, self.reward_size, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, self.env.action_space.n, self.reward_size, device=self.device)
        self.expected_values = torch.zeros(*shape, self.reward_size, device=self.device)
        self.rewards = torch.zeros(*shape, self.reward_size, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # initialize the pareto weights

        self.weights = torch.ones(shape[1], self.reward_size, device=self.device)/self.reward_size

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, self.reward_size, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, self.reward_size, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, eps=adam_eps)

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.model.recurrent:
                    value, memory = self.model(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    value = self.model(preprocessed_obs)
            action = self.pareto_action(value, self.weights)

            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.model.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)


            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i])#.item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i])#.item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

                    # reroll the weights for that episode
                    self.weights[i,0] = torch.rand(1)
                    self.weights[i,1] = 1-self.weights[i,0]

            self.log_episode_return = (self.log_episode_return.T*self.mask).T
            self.log_episode_reshaped_return = (self.log_episode_reshaped_return.T * self.mask).T
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.model.recurrent:
                next_value, _ = self.model(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                next_value = self.model(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_mask = torch.vstack([next_mask] * 2).T

            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value

            self.expected_values[i] = self.rewards[i] +  (self.pareto_rewards(next_value,self.weights) * (self.discount * next_mask))
 #           self.advantages[i] = delta + (next_advantage.T * (self.discount * self.gae_lambda *  next_mask)).T

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.model.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1,self.reward_size)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1,self.reward_size)
        exps.exp_value = self.expected_values.transpose(0, 1).reshape(-1,self.reward_size)
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    def update_parameters(self, exps):
        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Initialize memory

        if self.model.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Compute loss

            if self.model.recurrent:
                value, memory = self.model(sb.obs, memory * sb.mask)
            else:
                value = self.model(sb.obs)

#            entropy = dist.entropy().mean()

#            policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

            loss = (value - sb.exp_value.unsqueeze(1)).pow(2).mean()

            # Update batch values

            update_loss += loss

        # Update update values

        update_value /= self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.model.parameters()) ** 0.5
#        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = np.arange(0, self.num_frames, self.recurrence)
        return starting_indexes

    def pareto_action(self, values, weights):
        #col = torch.randint(0,self.reward_size,(1,))
        #return torch.max(values[:,:,col], dim=1).indices.squeeze()

        return torch.tensor(
                [torch.argmax(torch.matmul(values[i,:,:],weights[i,:]))
                 for i in range(values.shape[0])])

    def pareto_rewards(self, values, weights):
        #col = torch.randint(0,self.reward_size,(1,))
        #inds = torch.max(values[:,:,col], dim=1).indices
        #return values.gather(inds)

        actions = self.pareto_action(values, weights)
        return torch.vstack([values[i,actions[i],:] for i in range(values.shape[0])])