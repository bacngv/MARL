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

        for i in range(len(ids)):
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
        self._init_weights()

    def _construct_net(self):
        return nn.Sequential(
            nn.Linear(self.obs_space, 256),
            nn.LayerNorm(256),
            nn.ReLU(),  
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.act_space)
        )
    
    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        action_logits = self.net(obs)
        
        assert not torch.isnan(action_logits).any(), "NaNs found in actor network output"
    
        if available_actions is not None:
            action_logits = action_logits.masked_fill(~available_actions, float('-inf'))
        temperature = 1.0
        action_probs = F.softmax(action_logits / temperature, dim=-1)
        
        dist = torch.distributions.Categorical(action_probs)
        
        if deterministic:
            action_probs = F.softmax(action_logits / temperature, dim=-1)
            action_probs = torch.clamp(action_probs, min=1e-8, max=1.0) 
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        else:
            epsilon = 0.001
            if np.random.random() < epsilon:
                actions = torch.randint(0, self.act_space, (obs.shape[0],)).to(obs.device)
            else:
                actions = dist.sample()
            
        action_log_probs = dist.log_prob(actions)
        return actions, action_log_probs, rnn_states
class Critic(nn.Module):
    def __init__(self, cent_obs_space, device):
        super(Critic, self).__init__()
        self.device = device
        self.cent_obs_space = cent_obs_space
        self.net = self._construct_net()
        self._init_weights()

    def _construct_net(self):
        return nn.Sequential(
            nn.Linear(self.cent_obs_space, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
    
    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
    def forward(self, cent_obs, rnn_states, masks):
        values = self.net(cent_obs)
        return values, rnn_states
    
class TrainingLogger:
    def __init__(self):
        self.rewards = []
        self.losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.kl_divergences = []

    def log_reward(self, reward):
        self.rewards.append(reward)

    def log_loss(self, loss):
        self.losses.append(loss)

    def log_actor_loss(self, actor_loss):
        self.actor_losses.append(actor_loss)

    def log_critic_loss(self, critic_loss):
        self.critic_losses.append(critic_loss)

    def log_entropy_loss(self, entropy_loss):
        self.entropy_losses.append(entropy_loss)

    def log_kl_divergence(self, kl_divergence):
        self.kl_divergences.append(kl_divergence)

    def save_logs(self, file_path):
        logs = {
            'rewards': self.rewards,
            'losses': self.losses,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'entropy_losses': self.entropy_losses,
            'kl_divergences': self.kl_divergences
        }
        torch.save(logs, file_path)

    def load_logs(self, file_path):
        logs = torch.load(file_path)
        self.rewards = logs['rewards']
        self.losses = logs['losses']
        self.actor_losses = logs['actor_losses']
        self.critic_losses = logs['critic_losses']
        self.entropy_losses = logs['entropy_losses']
        self.kl_divergences = logs['kl_divergences']

class PPOPolicy(nn.Module):
    def __init__(self, env, name, handle, value_coef=0.5, ent_coef=0.01, gamma=0.99, 
                 batch_size=128,
                 learning_rate=1e-4,
                 use_cuda=False, 
                 clip_epsilon=1.0, 
                 ppo_epochs=4,
                 minibatch_size=32,
                 max_grad_norm=0.5,
                 gae_lambda=0.99,
                 reward_scaling=2.5):
        super(PPOPolicy, self).__init__()
        
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

        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.reward_scaling = reward_scaling
        self.view_buf = np.empty([1,] + list(self.view_space))
        self.feature_buf = np.empty([1,] + [self.feature_space])
        self.action_buf = np.empty(1, dtype=np.int32)
        self.reward_buf = np.empty(1, dtype=np.float32)
        self.log_probs_buf = np.empty(1, dtype=np.float32)
        self.replay_buffer = EpisodesBuffer()
        self.actor = Actor(np.prod(self.view_space) + self.feature_space, self.num_actions, device='cuda' if use_cuda else 'cpu')
        self.critic = Critic(np.prod(self.view_space) + self.feature_space, device='cuda' if use_cuda else 'cpu')
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

    def act(self, **kwargs):
        device = next(self.actor.parameters()).device
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
        values, _ = self.critic(combined_input, None, None)
        return values.detach().cpu().numpy()

    def train(self, cuda):
        batch_data = list(self.replay_buffer.episodes())
        self.replay_buffer.reset()
        n = sum(len(ep.rewards) for ep in batch_data)
        self._resize_buffers(n)
        
        view, feature, action, reward, log_probs = self._fill_buffers(batch_data)
        device = torch.device('cuda' if cuda else 'cpu')
        tensor_data = self._convert_to_tensors(view, feature, action, reward, log_probs)
        dataset = TensorDataset(*tensor_data)
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)
        
        # calculate advantages
        with torch.no_grad():
            combined_input = torch.cat([tensor_data[0].flatten(1), tensor_data[1]], dim=-1)
            old_values = self.critic(combined_input, None, None)[0].flatten()
            advantages = (tensor_data[3] - old_values).cpu().numpy()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.FloatTensor(advantages).to(device)

        # Initialize logger
        logger = TrainingLogger()

        for epoch in range(self.ppo_epochs):
            for batch in loader:
                stop_early = self._update_network(batch, advantages, logger)
                if stop_early:
                    break

        # Save logs after training
        logger.save_logs('training_logs.pth')

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
        all_returns = []
        all_advantages = []
        
        for episode in batch_data:
            m = len(episode.rewards)
            
            # calculate discounted returns and advantages
            returns, advantages = self._calculate_returns(episode)
            all_returns.extend(returns)
            all_advantages.extend(advantages)
            
            # fill buffers
            self.view_buf[ct:ct+m] = episode.views
            self.feature_buf[ct:ct+m] = episode.features
            self.action_buf[ct:ct+m] = episode.actions
            self.reward_buf[ct:ct+m] = returns
            if hasattr(episode, 'probs') and len(episode.probs) > 0:
                self.log_probs_buf[ct:ct+m] = np.log(episode.probs)
            else:
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
        """Calculate returns using GAE with modified reward scaling"""
        returns = np.zeros_like(episode.rewards, dtype=np.float32)
        advantages = np.zeros_like(episode.rewards, dtype=np.float32)
        
        # apply reward scaling and clipping
        scaled_rewards = np.clip(
            np.array(episode.rewards, dtype=np.float32) * self.reward_scaling,
            -10000.0, 10000.0  
        )
        with torch.no_grad():
            values = []
            for i in range(len(episode.views)):
                v = self._calc_value(episode.views[i], episode.features[i])[0]
                values.append(v)
            values = np.array(values, dtype=np.float32).flatten()
            
        # calculate GAE with normalization
        gae = 0
        for t in reversed(range(len(scaled_rewards))):
            if t == len(scaled_rewards) - 1:
                next_value = 0 if episode.terminal else values[t]  
            else:
                next_value = values[t + 1]
                
            delta = scaled_rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - episode.terminal) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return returns, advantages

    def _convert_to_tensors(self, view, feature, action, reward, log_probs):
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        return (
            torch.FloatTensor(view).to(device),
            torch.FloatTensor(feature).to(device),
            torch.LongTensor(action).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(log_probs).to(device)
        )

    def _update_network(self, batch, advantages, logger):
        batch_view, batch_feat, batch_act, batch_ret, batch_old_logp = batch
        batch_size = batch_view.shape[0]
        advantages = advantages[:batch_size]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        combined_input = torch.cat([batch_view.flatten(1), batch_feat], dim=-1)
        values, _ = self.critic(combined_input, None, None)
        values = values.squeeze(-1)
        value_pred_clipped = batch_old_logp + torch.clamp(
            values - batch_old_logp,
            -self.clip_epsilon,
            self.clip_epsilon
        )
        value_losses = (values - batch_ret).pow(2)
        value_losses_clipped = (value_pred_clipped - batch_ret).pow(2)
        value_loss = self.value_coef * torch.max(value_losses, value_losses_clipped).mean()
        dist = torch.distributions.Categorical(F.softmax(self.actor.net(combined_input), dim=-1))
        new_logp = dist.log_prob(batch_act)
        ratio = torch.exp(new_logp - batch_old_logp)
        ratio = torch.clamp(ratio, 0.0, 10.0) 
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy = dist.entropy().mean()
        entropy_loss = -self.ent_coef * entropy
        total_loss = policy_loss + value_loss + entropy_loss
        
        # Log losses
        logger.log_loss(total_loss.item())
        logger.log_actor_loss(policy_loss.item())
        logger.log_critic_loss(value_loss.item())
        logger.log_entropy_loss(entropy_loss.item())
        
        # optimize
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        
        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # calculate KL divergence for early stopping
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
        logger.log_kl_divergence(approx_kl)
        
        if approx_kl > 0.015:
            return True
            
        return False

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