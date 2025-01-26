import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import tools

class RNN(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args['rnn_hidden_dim'])
        self.rnn = nn.GRUCell(args['rnn_hidden_dim'], args['rnn_hidden_dim'])
        self.fc2 = nn.Linear(args['rnn_hidden_dim'], args['num_actions'])

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args['rnn_hidden_dim'])
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args

        if args['two_hyper_layers']:
            self.hyper_w1 = nn.Sequential(nn.Linear(args['state_shape'], args['hyper_hidden_dim']),
                                          nn.ReLU(),
                                          nn.Linear(args['hyper_hidden_dim'], args['n_agents'] * args['qmix_hidden_dim']))
            self.hyper_w2 = nn.Sequential(nn.Linear(args['state_shape'], args['hyper_hidden_dim']),
                                          nn.ReLU(),
                                          nn.Linear(args['hyper_hidden_dim'], args['qmix_hidden_dim']))
        else:
            self.hyper_w1 = nn.Linear(args['state_space'], args['num_agents'] * args['qmix_hidden_dim'])
            self.hyper_w2 = nn.Linear(args['state_space'], args['qmix_hidden_dim'] * 1)

        self.hyper_b1 = nn.Linear(args['state_space'], args['qmix_hidden_dim'])
        self.hyper_b2 = nn.Sequential(nn.Linear(args['state_space'], args['qmix_hidden_dim']),
                                      nn.ReLU(),
                                      nn.Linear(args['qmix_hidden_dim'], 1))

    def forward(self, q_values, states):
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args['num_agents'])
        states = states.reshape(-1, self.args['state_space'])

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(-1, self.args['num_agents'], self.args['qmix_hidden_dim'])
        b1 = b1.view(-1, 1, self.args['qmix_hidden_dim'])

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.args['qmix_hidden_dim'], 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total

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
        
        self.eval_rnn = RNN(input_shape, {
            'rnn_hidden_dim': rnn_hidden_dim, 
            'num_actions': self.num_actions
        })
        self.target_rnn = RNN(input_shape, {
            'rnn_hidden_dim': rnn_hidden_dim, 
            'num_actions': self.num_actions
        })
        
        self.eval_qmix_net = QMixNet({
            'state_space': self.state_space, 
            'num_agents': self.num_agents, 
            'qmix_hidden_dim': 32,  # example value, adjust as needed
            'two_hyper_layers': False
        })
        self.target_qmix_net = QMixNet({
            'state_space': self.state_space, 
            'num_agents': self.num_agents, 
            'qmix_hidden_dim': 32,  # example value, adjust as needed
            'two_hyper_layers': False
        })
        
        # Optimizer
        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        self.optim = torch.optim.RMSprop(self.eval_parameters, lr=learning_rate)
        
        # Replay Buffer
        self.replay_buffer = tools.ReplayBufferQMIX({
            'num_actions': self.num_actions,
            'num_agents': self.num_agents,
            'state_space': self.state_space,
            'obs_space': self.obs_space,
            'buffer_size': 5000,  # adjust as needed
            'max_episode_steps': 100  # adjust as needed
        })
        
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
        
        self.use_cuda = use_cuda
        
        # Hidden states
        self.eval_hidden = None
        self.target_hidden = None
    
    def train(self, train_step):
        # Sample from replay buffer
        batch_data = self.replay_buffer.sample(self.batch_size)
        
        # Prepare training data
        if not batch_data:
            return
        
        episode_num = batch_data['o'].shape[0]
        max_episode_len = batch_data['o'].shape[1]
        
        # Initialize hidden states
        self.init_hidden(episode_num)
        
        # Learn
        q_evals, q_targets = self.get_q_values(batch_data, max_episode_len)
        
        # Compute loss 
        u_onehot = torch.zeros(batch_data['u'].shape + (self.num_actions,))
        u_onehot.scatter_(-1, batch_data['u'][..., np.newaxis], 1)
        
        q_evals = torch.gather(q_evals, dim=3, index=u_onehot.long()).squeeze(3)
        
        q_targets = q_targets.max(dim=3)[0]
        
        q_total_eval = self.eval_qmix_net(q_evals, torch.FloatTensor(batch_data['s']))
        q_total_target = self.target_qmix_net(q_targets, torch.FloatTensor(batch_data['s']))
        
        targets = torch.FloatTensor(batch_data['r']) + self.gamma * q_total_target * (1 - torch.FloatTensor(batch_data['terminated']))
        
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
        inputs_next.append(batch['o_next'][:, transition_idx])
        
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
        # Prepare episode batch for ReplayBufferQMIX
        episode_batch = {
            'o': kwargs['state'][0],
            'u': kwargs['acts'].reshape(-1, self.num_agents, 1),
            's': kwargs.get('global_state', kwargs['state'][1]),
            'r': kwargs['rewards'].reshape(-1, self.num_agents),
            'o_next': kwargs.get('next_state', kwargs['state'][0]),
            's_next': kwargs.get('next_global_state', kwargs['state'][1]),
            'u_onehot': F.one_hot(torch.tensor(kwargs['acts']), num_classes=self.num_actions).numpy(),
            'padded': np.zeros((len(kwargs['acts']), 1)),
            'terminated': np.zeros((len(kwargs['acts']), 1))
        }
        self.replay_buffer.store_episode(episode_batch)
    
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