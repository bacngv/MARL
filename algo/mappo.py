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

def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

class Actor_RNN(nn.Module):
    def __init__(self, obs_space, act_space, device):
        super(Actor_RNN, self).__init__()
        self.device = device
        self.obs_space = obs_space
        self.act_space = act_space
        self.rnn_hidden_dim = 256
        self.fc_in = nn.Sequential(
            nn.Linear(obs_space, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.rnn = nn.GRUCell(256, self.rnn_hidden_dim)
        self.fc_out = nn.Linear(self.rnn_hidden_dim, act_space)
        orthogonal_init(self.fc_in[0])
        orthogonal_init(self.rnn, gain=1.0)
        orthogonal_init(self.fc_out, gain=0.01)

    def get_action_logits(self, obs):
        """Direct computation of action logits without RNN for training"""
        x = self.fc_in(obs)
        rnn_states = torch.zeros(obs.size(0), self.rnn_hidden_dim).to(self.device)
        rnn_out = self.rnn(x, rnn_states)
        return self.fc_out(rnn_out)

    def forward(self, obs, rnn_states, masks=None, available_actions=None, deterministic=False):
        batch_size = obs.size(0)
        x = self.fc_in(obs)
        if rnn_states is None:
            rnn_states = torch.zeros(batch_size, self.rnn_hidden_dim).to(self.device)
        if rnn_states.size(0) != batch_size:
            rnn_states = rnn_states[:batch_size]
            if rnn_states.size(0) < batch_size:
                padding = torch.zeros(batch_size - rnn_states.size(0), self.rnn_hidden_dim).to(self.device)
                rnn_states = torch.cat([rnn_states, padding], dim=0)
        if masks is not None:
            if masks.dim() == 1:
                masks = masks.unsqueeze(1)
            if masks.size(0) != batch_size:
                masks = masks[:batch_size]
            rnn_states = rnn_states * masks
        rnn_states = self.rnn(x, rnn_states)
        action_logits = self.fc_out(rnn_states)
        if available_actions is not None:
            action_logits = action_logits.masked_fill(~available_actions, float('-inf'))
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        if deterministic:
            actions = torch.argmax(action_probs, dim=-1)
        else:
            actions = dist.sample()
        action_log_probs = dist.log_prob(actions)
        return actions, action_log_probs, rnn_states

class CentralizedCritic_RNN(nn.Module):
    def __init__(self, cent_obs_dim, num_agents, device):
        super(CentralizedCritic_RNN, self).__init__()
        self.device = device
        self.num_agents = num_agents
        self.agent_obs_dim = cent_obs_dim
        self.cent_obs_dim = cent_obs_dim * num_agents
        self.rnn_hidden_dim = 256
        self.fc_in = nn.Sequential(
            nn.Linear(self.cent_obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        self.rnn = nn.GRUCell(512, self.rnn_hidden_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        orthogonal_init(self.fc_in[0])
        orthogonal_init(self.rnn, gain=1.0)
        orthogonal_init(self.fc_out[0])
        orthogonal_init(self.fc_out[3], gain=1.0)

    def forward(self, cent_obs, rnn_states, masks):
        batch_size = cent_obs.shape[0]
        if cent_obs.shape[1] == self.agent_obs_dim:
            cent_obs = cent_obs.unsqueeze(1).repeat(1, self.num_agents, 1)
            cent_obs = cent_obs.reshape(batch_size, -1)
        x = self.fc_in(cent_obs)
        if rnn_states is None:
            rnn_states = torch.zeros(batch_size, self.rnn_hidden_dim).to(self.device)
        if masks is not None:
            rnn_states = rnn_states * masks
        rnn_states = self.rnn(x, rnn_states)
        values = self.fc_out(rnn_states)
        return values, rnn_states

class MAPPOPolicy(nn.Module):
    def __init__(self, env, name, handle, num_agents, value_coef=0.5, ent_coef=0.01, gamma=0.99, 
                 batch_size=128, learning_rate=1e-4, use_cuda=False, clip_epsilon=0.2, 
                 ppo_epochs=4, minibatch_size=32, max_grad_norm=0.5,
                 gae_lambda=0.95, reward_scaling=2.5):
        super(MAPPOPolicy, self).__init__()
        
        self.env = env
        self.name = name
        self.num_agents = num_agents
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
        
        # Initialize separate buffers for each agent
        self.view_bufs = [np.empty([1,] + list(self.view_space)) for _ in range(num_agents)]
        self.feature_bufs = [np.empty([1,] + [self.feature_space]) for _ in range(num_agents)]
        self.action_bufs = [np.empty(1, dtype=np.int32) for _ in range(num_agents)]
        self.reward_bufs = [np.empty(1, dtype=np.float32) for _ in range(num_agents)]
        self.log_probs_bufs = [np.empty(1, dtype=np.float32) for _ in range(num_agents)]
        
        # Initialize separate replay buffers for each agent
        self.replay_buffers = [EpisodesBuffer() for _ in range(num_agents)]
        
        obs_dim = np.prod(self.view_space) + self.feature_space
        # Create separate actors for each agent but share the critic
        self.actors = nn.ModuleList([
            Actor_RNN(obs_dim, self.num_actions, device='cuda' if use_cuda else 'cpu')
            for _ in range(num_agents)
        ])
        self.critic = CentralizedCritic_RNN(obs_dim, num_agents, device='cuda' if use_cuda else 'cpu')
        
        # Separate optimizers for each actor
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=learning_rate)
            for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # RNN states for each agent
        self.actor_rnn_states = [None for _ in range(num_agents)]
        self.critic_rnn_states = None
        
        # Loss tracking for each agent
        self.current_policy_losses = [0 for _ in range(num_agents)]
        self.current_value_loss = 0
        self.current_entropy_losses = [0 for _ in range(num_agents)]
        self.current_total_losses = [0 for _ in range(num_agents)]
        self.loss_history = []
    def _convert_to_tensors(self, view, feature, action, reward, log_probs):
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        view_tensor = torch.FloatTensor(view).to(device)
        feature_tensor = torch.FloatTensor(feature).to(device)
        action_tensor = torch.LongTensor(action).to(device)
        reward_tensor = torch.FloatTensor(reward).to(device)
        log_probs_tensor = torch.FloatTensor(log_probs).to(device)
        
        return view_tensor, feature_tensor, action_tensor, reward_tensor, log_probs_tensor
    def flush_buffer(self, agent_idx, **kwargs):
        self.replay_buffers[agent_idx].push(**kwargs)

    def act(self, agent_idx, **kwargs):
        device = next(self.actors[agent_idx].parameters()).device
        if isinstance(kwargs['obs'], torch.Tensor):
            obs = kwargs['obs'].to(device)
            if len(obs.shape) == 4:
                obs = obs.flatten(1)
        else:
            obs = torch.FloatTensor(kwargs['obs']).to(device).flatten(1)
            
        if isinstance(kwargs['feature'], torch.Tensor):
            feature = kwargs['feature'].to(device)
        else:
            feature = torch.FloatTensor(kwargs['feature']).to(device)
            
        combined_input = torch.cat([obs, feature], dim=-1)
        batch_size = combined_input.size(0)
        masks = kwargs.get('masks', torch.ones(batch_size, 1).to(device))
        
        actions, action_log_probs, self.actor_rnn_states[agent_idx] = self.actors[agent_idx](
            combined_input,
            self.actor_rnn_states[agent_idx],
            masks,
            kwargs.get('available_actions', None),
            kwargs.get('deterministic', False)
        )
        
        if kwargs.get('return_log_prob', False):
            return actions.cpu().numpy().astype(np.int32), action_log_probs.detach().cpu().numpy()
        else:
            return actions.cpu().numpy().astype(np.int32)

    def _collect_centralized_obs(self, obs_batch, feature_batch):
        """Combine observations from all agents for centralized critic"""
        batch_size = obs_batch.shape[0]
        obs_dim = np.prod(self.view_space) + self.feature_space
        if len(obs_batch.shape) == 4: 
            obs_batch = obs_batch.unsqueeze(1)
            feature_batch = feature_batch.unsqueeze(1)
        # Khởi tạo tensor zeros để chứa observations của tất cả agents
        cent_obs = torch.zeros(batch_size, self.num_agents * obs_dim).to(obs_batch.device)
        
        # Chỉ điền data cho các agent còn sống
        for i in range(min(self.num_agents, obs_batch.shape[1])):
            start_idx = i * obs_dim
            end_idx = start_idx + obs_dim
            agent_obs = torch.cat([
                obs_batch[:, i].flatten(1),
                feature_batch[:, i]
            ], dim=-1)
            cent_obs[:, start_idx:end_idx] = agent_obs
        
        # Các agent đã chết sẽ giữ nguyên giá trị 0 trong cent_obs
        return cent_obs

    def train(self, cuda):
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        # Process each agent's data
        for agent_idx in range(self.num_agents):
            batch_data = list(self.replay_buffers[agent_idx].episodes())
            if not batch_data:
                continue
                
            self.replay_buffers[agent_idx].reset()
            n = sum(len(ep.rewards) for ep in batch_data)
            if n == 0:
                continue
                
            self._resize_buffers(n, agent_idx)
            buffer_data = self._fill_buffers(batch_data, agent_idx)
            if buffer_data is None:
                continue
                
            view, feature, action, reward, log_probs = buffer_data
            device = torch.device('cuda' if cuda else 'cpu')
            tensor_data = self._convert_to_tensors(view, feature, action, reward, log_probs)
            dataset = TensorDataset(*tensor_data)
            loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)
            
            with torch.no_grad():
                cent_obs = self._collect_centralized_obs(tensor_data[0], tensor_data[1])
                old_values = self.critic(cent_obs, None, None)[0].flatten()
                advantages = (tensor_data[3] - old_values).cpu().numpy()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = torch.FloatTensor(advantages).to(device)
                
            for _ in range(self.ppo_epochs):
                for batch in loader:
                    losses = self._update_network(batch, advantages, agent_idx)
                    total_policy_loss += losses['policy_loss']
                    total_value_loss += losses['value_loss']
                    total_entropy_loss += losses['entropy_loss']
        
        # Average losses across agents
        num_updates = self.num_agents * self.ppo_epochs
        if num_updates > 0:
            self.current_policy_loss = total_policy_loss / num_updates
            self.current_value_loss = total_value_loss / num_updates
            self.current_entropy_loss = total_entropy_loss / num_updates
            self.current_total_loss = (self.current_policy_loss + 
                                     self.current_value_loss + 
                                     self.current_entropy_loss)

    def _update_network(self, batch, advantages, agent_idx):
        batch_view, batch_feat, batch_act, batch_ret, batch_old_logp = batch
        batch_size = batch_view.shape[0]
        advantages = advantages[:batch_size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get centralized observations for critic
        cent_obs = self._collect_centralized_obs(batch_view, batch_feat)
        combined_input = torch.cat([batch_view.flatten(1), batch_feat], dim=-1)
        
        # Calculate value loss with centralized critic
        values, _ = self.critic(cent_obs, None, None)
        values = values.squeeze(-1)
        value_loss = self.value_coef * F.mse_loss(values, batch_ret)
        
        # Calculate policy loss for specific agent
        action_logits = self.actors[agent_idx].get_action_logits(combined_input)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        new_logp = dist.log_prob(batch_act)
        ratio = torch.exp(new_logp - batch_old_logp)
        ratio = torch.clamp(ratio, 0.0, 10.0)
        
        # PPO policy loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate entropy loss
        entropy = dist.entropy().mean()
        entropy_loss = -self.ent_coef * entropy
        
        # Calculate total loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        # Optimize
        self.actor_optimizers[agent_idx].zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        
        self.actor_optimizers[agent_idx].step()
        self.critic_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    def _resize_buffers(self, n, agent_idx):
        buffers = [
            self.view_bufs[agent_idx], 
            self.feature_bufs[agent_idx],
            self.action_bufs[agent_idx],
            self.reward_bufs[agent_idx],
            self.log_probs_bufs[agent_idx]
        ]
        shapes = [list(self.view_space), [self.feature_space], 1, 1, 1]
        
        for buf, shape in zip(buffers, shapes):
            if len(buf.shape) == 1:
                buf.resize(n, refcheck=False)
            else:
                buf.resize([n] + shape, refcheck=False)
    def _fill_buffers(self, batch_data, agent_idx):
        """Fill the buffers with episode data for a specific agent"""
        ct = 0
        
        for episode in batch_data:
            m = len(episode.rewards)
            returns, advantages = self._calculate_returns(episode)
            
            self.view_bufs[agent_idx][ct:ct+m] = episode.views
            self.feature_bufs[agent_idx][ct:ct+m] = episode.features
            self.action_bufs[agent_idx][ct:ct+m] = episode.actions
            self.reward_bufs[agent_idx][ct:ct+m] = returns
            
            if hasattr(episode, 'probs') and len(episode.probs) > 0:
                self.log_probs_bufs[agent_idx][ct:ct+m] = np.log(episode.probs)
            else:
                views = torch.FloatTensor(episode.views)
                features = torch.FloatTensor(episode.features)
                
                with torch.no_grad():
                    _, recalc_log_probs = self.act(
                        agent_idx=agent_idx,
                        obs=views,
                        feature=features,
                        return_log_prob=True
                    )
                self.log_probs_bufs[agent_idx][ct:ct+m] = recalc_log_probs
                
            ct += m
            
        return (
            self.view_bufs[agent_idx][:ct],
            self.feature_bufs[agent_idx][:ct],
            self.action_bufs[agent_idx][:ct],
            self.reward_bufs[agent_idx][:ct],
            self.log_probs_bufs[agent_idx][:ct]
        )
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
            combined_obs = []
            
            for i in range(len(episode.views)):
                view = torch.FloatTensor(episode.views[i])
                feature = torch.FloatTensor(episode.features[i])
                
                if len(view.shape) == 3:
                    view = view.unsqueeze(0)
                if len(feature.shape) == 1:
                    feature = feature.unsqueeze(0)
                    
                if self.use_cuda:
                    view = view.cuda()
                    feature = feature.cuda()
                    
                combined_input = torch.cat([view.flatten(1), feature], dim=-1)
                combined_obs.append(combined_input)
                
            combined_obs = torch.cat(combined_obs, dim=0)
            combined_obs = combined_obs.repeat(1, self.num_agents)
            masks = torch.ones(combined_obs.size(0), 1).to(combined_obs.device)
            
            values, _ = self.critic(combined_obs, None, masks)
            values = values.cpu().numpy().flatten()
                
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
    def _calc_value(self, obs, feature):
        if self.use_cuda:
            obs = torch.FloatTensor(obs).cuda().unsqueeze(0)
            feature = torch.FloatTensor(feature).cuda().unsqueeze(0)
        else:
            obs = torch.FloatTensor(obs).unsqueeze(0)
            feature = torch.FloatTensor(feature).unsqueeze(0)
            
        combined_input = torch.cat([obs.flatten(1), feature], dim=-1)
        combined_input = combined_input.repeat(1, self.num_agents)
        batch_size = combined_input.size(0)
        masks = torch.ones(batch_size, 1).to(combined_input.device)
        
        values, self.critic_rnn_states = self.critic(
            combined_input,
            self.critic_rnn_states,
            masks
        )
        return values.detach().cpu().numpy()

    def reset_rnn_states(self):
        """Reset RNN states for all agents at the beginning of new episodes"""
        self.actor_rnn_states = [None for _ in range(self.num_agents)]
        self.critic_rnn_states = None
    def get_loss(self):
        return {
            'policy_loss': self.current_policy_loss,
            'value_loss': self.current_value_loss,
            'entropy_loss': self.current_entropy_loss,
            'total_loss': self.current_total_loss
        }

    def get_loss_history(self):
        return self.loss_history

    def get_all_params(self):
        all_params = []
        for param in self.actors.parameters():
            if param.requires_grad:
                all_params.append(param)
        for param in self.critic.parameters():
            if param.requires_grad:
                all_params.append(param)
                
        return all_params

    def set_params(self, params):
        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        critic_params = [p for p in self.critic.parameters() if p.requires_grad]
        n_actor = len(actor_params)
        for param, new_param in zip(actor_params, params[:n_actor]):
            param.data.copy_(new_param.data)
        for param, new_param in zip(critic_params, params[n_actor:]):
            param.data.copy_(new_param.data)

    def get_param_values(self):
        params = []
        for param in self.get_all_params():
            params.append(param.data.cpu().numpy())
        return params

    def set_param_values(self, values):
        device = next(self.actor.parameters()).device
        params = []
        for value in values:
            params.append(torch.FloatTensor(value).to(device))
        self.set_params(params)

    def copy_params_from(self, source_policy):
        source_params = source_policy.get_all_params()
        self.set_params(source_params)

    def soft_update_from(self, source_policy, tau=0.01):
        target_params = self.get_all_params()
        source_params = source_policy.get_all_params()
        
        for target, source in zip(target_params, source_params):
            target.data.copy_(
                target.data * (1.0 - tau) + source.data * tau
            )
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"mappo_{step}")
        save_dict = {
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        # Save state dicts for all actors and their optimizers
        for i, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
            save_dict[f'actor_{i}_state_dict'] = actor.state_dict()
            save_dict[f'actor_optimizer_{i}_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, path)

    def load(self, dir_path, step=0):
        path = os.path.join(dir_path, f"mappo_{step}")
        checkpoint = torch.load(path)
        
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Load state dicts for all actors and their optimizers
        for i, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
            actor.load_state_dict(checkpoint[f'actor_{i}_state_dict'])
            optimizer.load_state_dict(checkpoint[f'actor_optimizer_{i}_state_dict'])