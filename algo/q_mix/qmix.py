import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import tools

class QMIX(nn.Module):
    def __init__(self, env, handle, name='QMIX', 
                 gamma=0.99, 
                 batch_size=64, 
                 learning_rate=1e-3, 
                 rnn_hidden_dim=64,
                 value_coef=0.1,
                 use_cuda=False,
                 last_action=False):
        super(QMIX, self).__init__()
        
        self.env = env
        self.name = name
        self.num_agents = env.unwrapped.env.get_agent_num(handle)
        self.num_actions = env.unwrapped.env.get_action_space(handle)[0]
        self.obs_space = env.unwrapped.env.get_obs_space(handle)[0]
        self.state_space = env.unwrapped.env.get_state_space()
        
        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rnn_hidden_dim = rnn_hidden_dim
        self.value_coef = value_coef
        self.last_action = last_action
        
        # Training buffers
        self.replay_buffer = tools.EpisodesBuffer()
        
        # Network construction
        self.net = self._construct_net()
        self.optim = torch.optim.Adam(self.get_all_params(), lr=self.learning_rate)
        self.use_cuda = use_cuda
        
    def _construct_net(self):
        net = nn.ModuleDict()
        
        # RNN for individual agent Q-values
        input_shape = self.obs_space + (self.num_actions if self.last_action else 0)
        net['rnn'] = nn.GRU(input_shape, self.rnn_hidden_dim, batch_first=True)
        net['q_head'] = nn.Linear(self.rnn_hidden_dim, self.num_actions)
        
        # QMIX mixing network
        net['hyper_w1'] = nn.Linear(self.state_space, self.rnn_hidden_dim * self.num_agents)
        net['hyper_w2'] = nn.Linear(self.state_space, 1)
        net['hyper_b1'] = nn.Linear(self.state_space, self.rnn_hidden_dim)
        
        return net
    
    def get_all_params(self):
        return list(self.net.parameters())
    
    def train(self, cuda):
        batch_data = self.replay_buffer.episodes()
        self.replay_buffer = tools.EpisodesBuffer()
        
        # Prepare training data
        states, obs, actions, rewards, next_states, next_obs = self._prepare_batch(batch_data)
        
        # Compute Q-values and target Q-values
        q_values = self._compute_q_values(obs, actions)
        target_q_values = self._compute_target_q_values(next_obs)
        
        # Compute TD targets
        td_targets = rewards + self.gamma * target_q_values
        
        # Compute loss
        loss = F.mse_loss(q_values, td_targets)
        
        # Optimize
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_all_params(), 5.0)
        self.optim.step()
        
        print(f'[*] QMIX Loss: {loss.item():.6f}')
    
    def _prepare_batch(self, batch_data):
        # Prepare training data from episode batches
        states, obs, actions, rewards, next_states, next_obs = [], [], [], [], [], []
        for episode in batch_data:
            states.append(episode.states)
            obs.append(episode.obs)
            actions.append(episode.actions)
            rewards.append(episode.rewards)
            next_states.append(episode.next_states)
            next_obs.append(episode.next_obs)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        obs = torch.FloatTensor(obs)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        next_obs = torch.FloatTensor(next_obs)
        
        return states, obs, actions, rewards, next_states, next_obs
    
    def _compute_q_values(self, obs, actions):
        # Compute individual agent Q-values
        h_0 = torch.zeros(1, obs.size(0), self.rnn_hidden_dim)
        _, h_n = self.net['rnn'](obs, h_0)
        q_individual = self.net['q_head'](h_n.squeeze(0))
        
        # QMIX mixing network
        q_total = self._mix_q_values(q_individual, obs)
        
        return q_total
    
    def _compute_target_q_values(self, next_obs):
        # Similar to _compute_q_values but with target computation
        h_0 = torch.zeros(1, next_obs.size(0), self.rnn_hidden_dim)
        _, h_n = self.net['rnn'](next_obs, h_0)
        q_individual = self.net['q_head'](h_n.squeeze(0))
        
        # Compute maximum Q-value
        q_max = q_individual.max(dim=-1)[0]
        
        return q_max
    
    def _mix_q_values(self, q_individual, state):
        # Hypernetwork for mixing individual Q-values
        w1 = torch.abs(self.net['hyper_w1'](state)).reshape(-1, self.num_agents, self.rnn_hidden_dim)
        w2 = torch.abs(self.net['hyper_w2'](state))
        b1 = self.net['hyper_b1'](state)
        
        # Mixing Q-values
        hidden = F.elu(torch.matmul(w1, q_individual.unsqueeze(-1)).squeeze(-1) + b1)
        q_total = torch.matmul(hidden, w2).squeeze(-1)
        
        return q_total
    
    def act(self, **kwargs):
        obs = kwargs['obs']
        h_0 = torch.zeros(1, obs.size(0), self.rnn_hidden_dim)
        _, h_n = self.net['rnn'](obs, h_0)
        q_values = self.net['q_head'](h_n.squeeze(0))
        
        action = q_values.argmax(dim=-1).numpy().astype(np.int32)
        return action.reshape((-1,))
    
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)
    
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"qmix_{step}")
        torch.save(self.net.state_dict(), file_path)
        print("[*] Model saved")
    
    def load(self, dir_path, step=0):
        file_path = os.path.join(dir_path, f"qmix_{step}")
        self.net.load_state_dict(torch.load(file_path))
        print("[*] Loaded model")