import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class EpisodesBufferEntry:
    """Entry for episode buffer"""
    def __init__(self):
        self.views = []
        self.features = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.terminal = False

    def append(self, view, feature, action, reward, alive, probs=None):
        self.views.append(view.copy())
        self.features.append(feature.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        if probs is not None:
            self.probs.append(probs)
        if not alive:
            self.terminal = True
            
class EpisodesBuffer:
    """Replay buffer to store a whole episode for all agents
       one entry for one agent
    """
    def __init__(self, use_mean=False):
        self.buffer = {}
        self.use_mean = use_mean

    def push(self, **kwargs):
        view, feature = kwargs['state']
        acts = kwargs['acts']
        rewards = kwargs['rewards']
        alives = kwargs['alives']
        ids = kwargs['ids']

        if self.use_mean:
            probs = kwargs['prob']

        buffer = self.buffer
        index = np.random.permutation(len(view))

        for i in range(len(ids)):
            i = index[i]
            entry = buffer.get(ids[i])
            if entry is None:
                entry = EpisodesBufferEntry()
                buffer[ids[i]] = entry

            if self.use_mean:
                entry.append(view[i], feature[i], acts[i], rewards[i], alives[i], probs=probs[i])
            else:
                entry.append(view[i], feature[i], acts[i], rewards[i], alives[i])

    def reset(self):
        """ clear replay buffer """
        self.buffer = {}

    def episodes(self):
        """ get episodes """
        return self.buffer.values()

class Actor(nn.Module):
    def __init__(self, obs_space, act_space, device):
        super(Actor, self).__init__()
        self.device = device
        self.obs_space = obs_space
        self.act_space = act_space
        self.net = self._construct_net()

    def _construct_net(self):
        # Define your actor network here
        net = nn.Sequential(
            nn.Linear(self.obs_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.act_space),
            nn.Softmax(dim=-1)
        )
        return net

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        # Forward pass through the actor network
        action_probs = self.net(obs)
        dist = torch.distributions.Categorical(action_probs)
        if deterministic:
            actions = torch.argmax(action_probs, dim=-1)
        else:
            actions = dist.sample()
        action_log_probs = dist.log_prob(actions)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        action_probs = self.net(obs)
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()
        return action_log_probs, dist_entropy


class Critic(nn.Module):
    def __init__(self, cent_obs_space, device):
        super(Critic, self).__init__()
        self.device = device
        self.cent_obs_space = cent_obs_space
        self.net = self._construct_net()

    def _construct_net(self):
        # Define your critic network here
        net = nn.Sequential(
            nn.Linear(self.cent_obs_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        return net

    def forward(self, cent_obs, rnn_states, masks):
        # Forward pass through the critic network
        values = self.net(cent_obs)
        return values, rnn_states

class MAPPOPolicy(nn.Module):
    def __init__(self, env, name, handle, value_coef=0.2, ent_coef=0.2, gamma=0.95, 
                 batch_size=128, learning_rate=3e-4, use_cuda=False, clip_epsilon=0.5, 
                 ppo_epochs=2, minibatch_size=64):
        super(MAPPOPolicy, self).__init__()
        
        self.env = env
        self.name = name
        self.view_space = env.unwrapped.env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.unwrapped.env.get_feature_space(handle)[0]
        self.num_actions = env.unwrapped.env.get_action_space(handle)[0]
        self.gamma = gamma
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.value_coef = value_coef
        self.ent_coef = ent_coef
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.use_cuda = use_cuda

        # Initialize buffers
        self.view_buf = np.empty([1,] + list(self.view_space))
        self.feature_buf = np.empty([1,] + [self.feature_space])
        self.action_buf = np.empty(1, dtype=np.int32)
        self.reward_buf = np.empty(1, dtype=np.float32)
        self.log_probs_buf = np.empty(1, dtype=np.float32)
        self.replay_buffer = EpisodesBuffer()

        # Initialize actor and critic
        self.actor = Actor(np.prod(self.view_space) + self.feature_space, self.num_actions, device='cuda' if use_cuda else 'cpu')
        self.critic = Critic(np.prod(self.view_space) + self.feature_space, device='cuda' if use_cuda else 'cpu')

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

    def act(self, **kwargs):
        # Get device 
        device = next(self.actor.parameters()).device
        
        # Move tensors to correct device
        if isinstance(kwargs['obs'], torch.Tensor):
            obs = kwargs['obs'].to(device).flatten(1)
        else:
            obs = torch.FloatTensor(kwargs['obs']).to(device).flatten(1)
            
        if isinstance(kwargs['feature'], torch.Tensor):
            feature = kwargs['feature'].to(device)
        else:
            feature = torch.FloatTensor(kwargs['feature']).to(device)
        
        combined_input = torch.cat([obs, feature], dim=-1)
        
        actions, action_log_probs, _ = self.actor(combined_input, None, None)
        
        # Return only the action array for environment interaction
        if kwargs.get('return_log_prob', False):
            return actions.cpu().numpy().astype(np.int32), action_log_probs.detach().cpu().numpy()
        else:
            return actions.cpu().numpy().astype(np.int32)

    def _calc_value(self, obs, feature):
        """
        Calculate the value of a given observation and feature.
        
        Args:
            obs (np.ndarray): Observation array.
            feature (np.ndarray): Feature array.
        
        Returns:
            np.ndarray: Value of the observation and feature.
        """
        if self.use_cuda:
            obs = torch.FloatTensor(obs).cuda().unsqueeze(0)
            feature = torch.FloatTensor(feature).cuda().unsqueeze(0)
        else:
            obs = torch.FloatTensor(obs).unsqueeze(0)
            feature = torch.FloatTensor(feature).unsqueeze(0)
        
        combined_input = torch.cat([obs.flatten(1), feature], dim=-1)
        
        # Forward pass through the critic network
        values, _ = self.critic(combined_input, None, None)
        return values.detach().cpu().numpy()

    def train(self, cuda):
        batch_data = list(self.replay_buffer.episodes())
        self.replay_buffer.reset()
        
        # Process episodes
        n = sum(len(ep.rewards) for ep in batch_data)
        self._resize_buffers(n)
        
        view, feature, action, reward, log_probs = self._fill_buffers(batch_data)
        
        # Convert to tensors and move to appropriate device
        device = torch.device('cuda' if cuda else 'cpu')
        tensor_data = self._convert_to_tensors(view, feature, action, reward, log_probs)
        dataset = TensorDataset(*tensor_data)
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)
        
        # Calculate advantages
        with torch.no_grad():
            combined_input = torch.cat([tensor_data[0].flatten(1), tensor_data[1]], dim=-1)
            old_values = self.critic(combined_input, None, None)[0].flatten()
            advantages = (tensor_data[3] - old_values).cpu().numpy()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.FloatTensor(advantages).to(device)

        # Training loop
        for _ in range(self.ppo_epochs):
            for batch in loader:
                self._update_network(batch, advantages)

    def _resize_buffers(self, n):
        buffers = [self.view_buf, self.feature_buf, self.action_buf, 
                   self.reward_buf, self.log_probs_buf]
        shapes = [list(self.view_space), [self.feature_space], 1, 1, 1]
        
        for buf, shape in zip(buffers, shapes):
            if len(buf.shape) == 1:
                buf.resize(n, refcheck=False)
            else:
                buf.resize([n] + shape, refcheck=False)

    def _fill_buffers(self, batch_data):
        ct = 0
        for episode in batch_data:
            m = len(episode.rewards)
            
            # Calculate discounted returns
            returns = self._calculate_returns(episode)
            
            # Fill buffers
            self.view_buf[ct:ct+m] = episode.views
            self.feature_buf[ct:ct+m] = episode.features
            self.action_buf[ct:ct+m] = episode.actions
            self.reward_buf[ct:ct+m] = returns
            
            # Convert probs to log_probs if they exist
            if hasattr(episode, 'probs') and len(episode.probs) > 0:
                self.log_probs_buf[ct:ct+m] = np.log(episode.probs)
            else:
                # If no probs available, initialize with zeros or recalculate
                views = torch.FloatTensor(episode.views)
                features = torch.FloatTensor(episode.features)
                
                with torch.no_grad():
                    _, recalc_log_probs = self.act(
                        obs=views,
                        feature=features,
                        return_log_prob=True
                    )
                self.log_probs_buf[ct:ct+m] = recalc_log_probs
            ct += m
            
        return (self.view_buf, self.feature_buf, self.action_buf, 
                self.reward_buf, self.log_probs_buf)

    def _calculate_returns(self, episode):
        returns = np.zeros_like(episode.rewards)
        R = self._calc_value(obs=episode.views[-1], feature=episode.features[-1])[0]
        
        for i in reversed(range(len(episode.rewards))):
            R = episode.rewards[i] + self.gamma * R
            returns[i] = R
        return returns

    def _convert_to_tensors(self, view, feature, action, reward, log_probs):
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        return (
            torch.FloatTensor(view).to(device),
            torch.FloatTensor(feature).to(device),
            torch.LongTensor(action).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(log_probs).to(device)
        )

    def _update_network(self, batch, advantages):
        batch_view, batch_feat, batch_act, batch_ret, batch_old_logp = batch
        batch_size = batch_view.shape[0]
        advantages = advantages[:batch_size]  # Make sure to use correct slice of advantages
        
        combined_input = torch.cat([batch_view.flatten(1), batch_feat], dim=-1)
        
        # Forward pass through actor and critic
        actions, action_log_probs, _ = self.actor(combined_input, None, None)
        values, _ = self.critic(combined_input, None, None)
        
        # Calculate losses
        dist = torch.distributions.Categorical(F.softmax(self.actor.net(combined_input), dim=-1))
        new_logp = dist.log_prob(batch_act)
        
        ratio = (new_logp - batch_old_logp).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = self.value_coef * (batch_ret - values).pow(2).mean()
        entropy_loss = -self.ent_coef * dist.entropy().mean()
        
        total_loss = policy_loss + value_loss + entropy_loss
        
        # Optimize
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"mappo_{step}")
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    def get_all_params(self):
        """Return all parameters of both actor and critic networks."""
        return list(self.actor.parameters()) + list(self.critic.parameters())

    def load(self, dir_path, step=0):
        path = os.path.join(dir_path, f"mappo_{step}")
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])