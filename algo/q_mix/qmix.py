import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qmix_net import RNN, QMixNet

from . import tools

class QMIX(nn.Module):
    def __init__(self, env, name, handle, gamma=0.99, batch_size=64, 
                 learning_rate=1e-3, use_cuda=False, 
                 last_action=False, rnn_hidden_dim=64, 
                 grad_norm_clip=10, target_update_cycle=200):
        super(QMIX, self).__init__()
        
        self.env = env
        self.name = name
        self.num_agents = env.unwrapped.env.get_agent_num(handle)
        self.num_actions = env.unwrapped.env.get_action_space(handle)[0]
        self.state_space = env.unwrapped.env.get_state_space()
        self.obs_space = env.unwrapped.env.get_obs_space(handle)[0]
        
        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.last_action = last_action
        self.rnn_hidden_dim = rnn_hidden_dim
        self.grad_norm_clip = grad_norm_clip
        self.target_update_cycle = target_update_cycle
        
        # Network initialization
        input_shape = self.obs_space
        if last_action:
            input_shape += self.num_actions
        
        self.eval_rnn = RNN(input_shape, {'rnn_hidden_dim': rnn_hidden_dim})
        self.target_rnn = RNN(input_shape, {'rnn_hidden_dim': rnn_hidden_dim})
        self.eval_qmix_net = QMixNet({'state_space': self.state_space})
        self.target_qmix_net = QMixNet({'state_space': self.state_space})
        
        # Optimizer
        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        self.optim = torch.optim.RMSprop(self.eval_parameters, lr=learning_rate)
        
        # Device configuration
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        if use_cuda:
            self.eval_rnn.to(self.device)
            self.target_rnn.to(self.device)
            self.eval_qmix_net.to(self.device)
            self.target_qmix_net.to(self.device)
        
        # Synchronize target networks
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        
        self.replay_buffer = tools.EpisodesBuffer()
        self.use_cuda = use_cuda
        
        # Hidden states
        self.eval_hidden = None
        self.target_hidden = None
    
    def train(self, train_step):
        # Sample from replay buffer
        batch_data = self.replay_buffer.episodes()
        self.replay_buffer = tools.EpisodesBuffer()
        
        # Prepare training data
        if not batch_data:
            return
        
        episode_num = batch_data[0].observations.shape[0]
        max_episode_len = batch_data[0].observations.shape[1]
        
        # Prepare batch dictionary
        batch = {
            'o': np.array([episode.observations for episode in batch_data]),
            'u': np.array([episode.actions for episode in batch_data]),
            'r': np.array([episode.rewards for episode in batch_data]),
            's': np.array([episode.states for episode in batch_data]),
            'terminated': np.zeros((episode_num, max_episode_len)),
            'padded': np.zeros((episode_num, max_episode_len)),
            'avail_u': np.ones((episode_num, max_episode_len, self.num_agents, self.num_actions)),
            'avail_u_next': np.ones((episode_num, max_episode_len, self.num_agents, self.num_actions))
        }
        
        # Initialize hidden states
        self.init_hidden(episode_num)
        
        # Learn
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        
        # Compute loss similar to original QMIX implementation
        u_onehot = torch.zeros(batch['u'].shape + (self.num_actions,))
        u_onehot.scatter_(-1, batch['u'][..., np.newaxis], 1)
        
        q_evals = torch.gather(q_evals, dim=3, index=u_onehot.long()).squeeze(3)
        
        q_targets = q_targets.max(dim=3)[0]
        
        q_total_eval = self.eval_qmix_net(q_evals, torch.FloatTensor(batch['s']))
        q_total_target = self.target_qmix_net(q_targets, torch.FloatTensor(batch['s']))
        
        targets = torch.FloatTensor(batch['r']) + self.gamma * q_total_target * (1 - torch.FloatTensor(batch['terminated']))
        
        loss = F.mse_loss(q_total_eval, targets.detach())
        
        # Optimize
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.grad_norm_clip)
        self.optim.step()
        
        # Update target networks periodically
        if train_step > 0 and train_step % self.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        
        return loss.item()
    
    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
            
            q_eval = q_eval.view(episode_num, self.num_agents, -1)
            q_target = q_target.view(episode_num, self.num_agents, -1)
            
            q_evals.append(q_eval)
            q_targets.append(q_target)
        
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        
        return q_evals, q_targets
    
    def _get_inputs(self, batch, transition_idx):
        obs = batch['o'][:, transition_idx]
        u_onehot = batch['u']
        episode_num = obs.shape[0]
        
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(batch['o'][:, transition_idx])
        
        if self.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        
        inputs = torch.cat([x.reshape(episode_num * self.num_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.num_agents, -1) for x in inputs_next], dim=1)
        
        return inputs, inputs_next
    
    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.num_agents, self.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.num_agents, self.rnn_hidden_dim))
    
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)
    
    def act(self, **kwargs):
        obs = kwargs['obs']
        self.eval_hidden = torch.zeros((1, self.num_agents, self.rnn_hidden_dim))
        q_values, _ = self.eval_rnn(obs, self.eval_hidden)
        action = q_values.argmax(dim=1).detach().cpu().numpy()
        return action.astype(np.int32).reshape((-1,))
    
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.eval_qmix_net.state_dict(), os.path.join(dir_path, f'{step}_qmix_net.pkl'))
        torch.save(self.eval_rnn.state_dict(), os.path.join(dir_path, f'{step}_rnn_net.pkl'))
    
    def load(self, dir_path, step=0):
        self.eval_qmix_net.load_state_dict(torch.load(os.path.join(dir_path, f'{step}_qmix_net.pkl')))
        self.eval_rnn.load_state_dict(torch.load(os.path.join(dir_path, f'{step}_rnn_net.pkl')))