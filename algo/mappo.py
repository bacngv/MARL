import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from . import tools

class MAPPOPolicy(nn.Module):
    def __init__(self, env, name, handle, value_coef=0.1, ent_coef=0.01, gamma=0.95, 
                 batch_size=128, learning_rate=3e-4, use_cuda=False, clip_epsilon=0.2, 
                 ppo_epochs=4, minibatch_size=32):
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
        self.replay_buffer = tools.EpisodesBuffer()

        self.net = self._construct_net()
        self.optim = torch.optim.Adam(self.get_all_params(), lr=learning_rate)

    def _construct_net(self):
        net = nn.ModuleDict({
            'obs_linear': nn.Linear(np.prod(self.view_space), 256),
            'emb_linear': nn.Linear(self.feature_space, 256),
            'cat_linear': nn.Linear(512, 512),
            'policy_linear': nn.Linear(512, self.num_actions),
            'value_linear': nn.Linear(512, 1)
        })
        return net

    def get_all_params(self):
        return self.net.parameters()

    def act(self, **kwargs):
        # Get device 
        device = next(self.net.parameters()).device
        
        # Move tensors to correct device
        if isinstance(kwargs['obs'], torch.Tensor):
            obs = kwargs['obs'].to(device).flatten(1)
        else:
            obs = torch.FloatTensor(kwargs['obs']).to(device).flatten(1)
            
        if isinstance(kwargs['feature'], torch.Tensor):
            feature = kwargs['feature'].to(device)
        else:
            feature = torch.FloatTensor(kwargs['feature']).to(device)
        
        h_view = F.relu(self.net['obs_linear'](obs))
        h_emb = F.relu(self.net['emb_linear'](feature))
        dense = torch.cat([h_view, h_emb], dim=-1)
        dense = F.relu(self.net['cat_linear'](dense))
        
        policy = F.softmax(self.net['policy_linear'](dense / 0.5), dim=-1)
        policy = torch.clamp(policy, 1e-10, 1-1e-10)
        
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return only the action array for environment interaction
        if kwargs.get('return_log_prob', False):
            return action.cpu().numpy().astype(np.int32), log_prob.detach().cpu().numpy()
        else:
            return action.cpu().numpy().astype(np.int32)

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
        
        # Flatten the observation
        flatten_view = obs.reshape(obs.size()[0], -1)
        
        # Forward pass through the network
        h_view = F.relu(self.net['obs_linear'](flatten_view))
        h_emb = F.relu(self.net['emb_linear'](feature))
        dense = torch.cat([h_view, h_emb], dim=-1)
        dense = F.relu(self.net['cat_linear'](dense))
        
        # Calculate the value
        value = self.net['value_linear'](dense)
        value = value.flatten()
        
        return value.detach().cpu().numpy()

    def train(self, cuda):
        batch_data = self.replay_buffer.episodes()
        self.replay_buffer = tools.EpisodesBuffer()
        
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
            old_values = self._calculate_values(tensor_data[0], tensor_data[1]).flatten()
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

    def _calculate_values(self, view, feature):
        """Calculate values for given observations and features.
        
        Args:
            view: Tensor of shape [batch_size, *view_space]
            feature: Tensor of shape [batch_size, feature_space]
        """
        device = next(self.net.parameters()).device
        
        # Ensure inputs are tensors and on correct device
        if not isinstance(view, torch.Tensor):
            view = torch.FloatTensor(view)
        if not isinstance(feature, torch.Tensor):
            feature = torch.FloatTensor(feature)
            
        view = view.to(device)
        feature = feature.to(device)
        
        # Reshape view tensor properly
        batch_size = view.shape[0]
        view_flat = view.reshape(batch_size, -1)
        
        # Forward pass
        h_view = F.relu(self.net['obs_linear'](view_flat))
        h_emb = F.relu(self.net['emb_linear'](feature))
        dense = F.relu(self.net['cat_linear'](torch.cat([h_view, h_emb], -1)))
        return self.net['value_linear'](dense)

    def _update_network(self, batch, advantages):
        batch_view, batch_feat, batch_act, batch_ret, batch_old_logp = batch
        batch_size = batch_view.shape[0]
        advantages = advantages[:batch_size]  # Make sure to use correct slice of advantages
        
        # Reshape view tensor
        view_flat = batch_view.reshape(batch_size, -1)
        
        # Forward pass
        h_view = F.relu(self.net['obs_linear'](view_flat))
        h_emb = F.relu(self.net['emb_linear'](batch_feat))
        dense = F.relu(self.net['cat_linear'](torch.cat([h_view, h_emb], -1)))
        
        policy = F.softmax(self.net['policy_linear'](dense), dim=-1)
        values = self.net['value_linear'](dense).flatten()
        
        # Calculate losses
        dist = torch.distributions.Categorical(policy)
        new_logp = dist.log_prob(batch_act)
        
        ratio = (new_logp - batch_old_logp).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = self.value_coef * (batch_ret - values).pow(2).mean()
        entropy_loss = -self.ent_coef * (policy * torch.log(policy)).sum(-1).mean()
        
        total_loss = policy_loss + value_loss + entropy_loss
        
        # Optimize
        self.optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optim.step()

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"mappo_{step}")
        torch.save(self.net.state_dict(), path)

    def load(self, dir_path, step=0):
        path = os.path.join(dir_path, f"mappo_{step}")
        self.net.load_state_dict(torch.load(path))