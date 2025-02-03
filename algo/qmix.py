import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .tools import QMixReplayBuffer
from . import base


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        # Assuming input_shape is (channels, height, width)
        input_size = input_shape[0] * input_shape[1]
        
        self.fc1 = nn.Linear(input_size, args['rnn_hidden_dim'])
        self.rnn = nn.GRUCell(args['rnn_hidden_dim'], args['rnn_hidden_dim'])
        self.fc2 = nn.Linear(args['rnn_hidden_dim'], args['num_actions'])
        
    def forward(self, inputs, hidden_state):
        # Print input shape for debugging
        #print(f"Input shape: {inputs.shape}")
        
        batch_size, channels, height, width= inputs.shape
        
        # Reshape inputs to (batch_size * num_agents, channels * height * width)
        x = inputs.reshape(batch_size * channels, height * width)
        #print(f"Reshaped x shape: {x.shape}")
        
        # Apply first layer
        x = F.relu(self.fc1(x))
        
        # Prepare hidden state
        h_in = hidden_state.reshape(batch_size * channels, -1)
        
        # Process through RNN
        h = self.rnn(x, h_in)
        
        # Reshape hidden state
        h = h.reshape(batch_size, channels, -1)
        
        # Get Q-values
        q = self.fc2(h)
        
        return q, h

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        
        self.hyper_w1 = nn.Linear(args['state_space'], args['num_agents'] * args['qmix_hidden_dim'])
        self.hyper_w2 = nn.Linear(args['state_space'], args['qmix_hidden_dim'])
        
        self.hyper_b1 = nn.Linear(args['state_space'], args['qmix_hidden_dim'])
        self.hyper_b2 = nn.Linear(args['state_space'], 1)

    def forward(self, agent_qs, states):
        batch_size = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.args['num_agents'])
        states = states.reshape(-1, self.args['state_space'])

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.args['num_agents'], self.args['qmix_hidden_dim'])
        b1 = b1.view(-1, 1, self.args['qmix_hidden_dim'])
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, self.args['qmix_hidden_dim'], 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(batch_size, -1, 1)

class QMIX(base.ValueNet):
    def __init__(self, env, name, handle, use_cuda=False, memory_size=2**10, batch_size=64, 
                 update_every=5, learning_rate=1e-4, tau=0.005, gamma=0.95):
        # Previous initialization code remains the same
        super().__init__(env, name, handle, update_every=update_every,
                        learning_rate=learning_rate, tau=tau, gamma=gamma)
        
        self.args = {
            'num_actions': 21,
            'num_agents': 64*2,
            'state_space': 45*45*5,
            'obs_space': 845,
            'rnn_hidden_dim': 32,
            'qmix_hidden_dim': 32,
            'batch_size': batch_size,
            'buffer_size': memory_size,
            'max_episode_steps': 300
        }
        
        # Previous initialization code remains the same...
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        self.eval_rnn = RNNAgent(self.view_space, self.args).to(self.device)
        self.target_rnn = RNNAgent(self.view_space, self.args).to(self.device)
        self.eval_qmix = QMixNet(self.args).to(self.device)
        self.target_qmix = QMixNet(self.args).to(self.device)
        
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix.load_state_dict(self.eval_qmix.state_dict())
        
        self.params = list(self.eval_rnn.parameters()) + list(self.eval_qmix.parameters())
        self.optimizer = torch.optim.RMSprop(self.params, lr=learning_rate)
        
        self.replay_buffer = QMixReplayBuffer(self.args)
        self.hidden_states = None
        
        # Add episode buffer to collect transitions before flushing
        self.episode_buffer = {
            'o': [],
            'u': [],
            's': [],
            'r': [],
            'o_next': [],
            's_next': [],
            'avail_u': [],
            'avail_u_next': [],
            'u_onehot': [],
            'padded': [],
            'terminated': []
        }

    def flush_buffer(self, **kwargs):
        """
        Flush the current episode buffer to the replay buffer.
        """
        # Convert lists to numpy arrays and reshape according to buffer requirements
        episode_batch = {}
        for key in kwargs.keys():
            if key in self.episode_buffer:
                self.episode_buffer[key].append(kwargs[key])
        
        # When episode is done, convert to numpy arrays and store
        if kwargs.get('terminated', False):
            for key in self.episode_buffer.keys():
                if len(self.episode_buffer[key]) > 0:
                    episode_batch[key] = np.array(self.episode_buffer[key])
            
            # Store the episode in the replay buffer
            if episode_batch:
                self.replay_buffer.store_episode(episode_batch)
            
            # Clear episode buffer
            for key in self.episode_buffer.keys():
                self.episode_buffer[key] = []

    def reset_buffer(self):
        """
        Reset the episode buffer.
        """
        for key in self.episode_buffer.keys():
            self.episode_buffer[key] = []

    def store_transition(self, **kwargs):
        self.replay_buffer.store_episode(kwargs)
        
    def train(self, cuda=False):
        if self.replay_buffer.current_size < self.args['batch_size']:
            return
            
        batch = self.replay_buffer.sample(self.args['batch_size'])
        
        # Convert batch to tensor and move to device if using cuda
        device = torch.device('cuda') if cuda else torch.device('cpu')
        
        obs = torch.FloatTensor(batch['o']).to(device)
        obs_next = torch.FloatTensor(batch['o_next']).to(device)
        states = torch.FloatTensor(batch['s']).to(device)
        states_next = torch.FloatTensor(batch['s_next']).to(device)
        actions = torch.LongTensor(batch['u']).to(device)
        rewards = torch.FloatTensor(batch['r']).to(device)
        terminated = torch.FloatTensor(batch['terminated']).to(device)
        mask = 1 - torch.FloatTensor(batch['padded']).to(device)
        
        # Reshape obs to match the expected input shape
        batch_size, max_episode_steps, num_agents, channels, height, width = obs.shape
        obs = obs.reshape(batch_size * max_episode_steps, num_agents, channels, height, width)
        obs_next = obs_next.reshape(batch_size * max_episode_steps, num_agents, channels, height, width)
        
        # Initialize hidden states
        self.init_hidden(batch_size * max_episode_steps)
        self.hidden_states = self.hidden_states.to(device)
        
        # Get Q-values for all agents
        q_evals = []
        q_targets = []
        
        for t in range(self.args['max_episode_steps']):
            inputs = obs[:, t]
            inputs_next = obs_next[:, t]
            
            q_eval, self.hidden_states = self.eval_rnn(inputs, self.hidden_states)
            q_target, _ = self.target_rnn(inputs_next, self.hidden_states)
            
            q_evals.append(q_eval)
            q_targets.append(q_target)
            
        # Stack Q-values
        q_evals = torch.stack(q_evals, dim=1)  # [batch_size, max_episode_len, n_agents, n_actions]
        q_targets = torch.stack(q_targets, dim=1)
        
        # Get chosen action Q-values
        chosen_action_qvals = torch.gather(q_evals, dim=3, index=actions).squeeze(3)
        
        # Mix Q-values
        chosen_action_qvals = chosen_action_qvals * mask
        target_max_qvals = q_targets.max(dim=3)[0] * mask
        
        q_total_eval = self.eval_qmix(chosen_action_qvals, states)
        q_total_target = self.target_qmix(target_max_qvals, states_next)
        
        targets = rewards + self.gamma * (1 - terminated) * q_total_target
        
        # Calculate loss and optimize
        td_error = (q_total_eval - targets.detach())
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimizer.step()
        
        if self.update_cnt % self.update_every == 0:
            self._update_target_network()
            
        self.update_cnt += 1
        
        return loss.item()
    
    def init_hidden(self, batch_size):
        # Create hidden states for each agent
        self.hidden_states = torch.zeros(batch_size, self.args['num_agents'], 
                                        self.args['rnn_hidden_dim'], 
                                        device=self.device)

    def act(self, obs, feature, prob, eps=0):
        # Ensure obs is on the correct device
        obs = obs.to(self.device)
        batch_size, num_agents, height, width = obs.shape
        
        # Initialize hidden states with correct batch size and number of agents
        self.hidden_states = torch.zeros(batch_size, num_agents, 
                                        self.args['rnn_hidden_dim'], 
                                        device=self.device)
        
        # Compute Q-values for each agent
        q_values, self.hidden_states = self.eval_rnn(obs, self.hidden_states)
        
        # Action selection
        actions = []
        for i in range(batch_size):
            for j in range(num_agents):
                if np.random.rand() < eps:
                    action = np.random.randint(0, self.args['num_actions'])
                else:
                    action = q_values[i, j].argmax().item()
                actions.append(action)
        
        return np.array(actions).reshape(batch_size, num_agents)

    def _update_target_network(self):
        for param, target_param in zip(self.eval_rnn.parameters(), 
                                     self.target_rnn.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                  (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.eval_qmix.parameters(),
                                     self.target_qmix.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                  (1 - self.tau) * target_param.data)
    
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        
        torch.save({
            'rnn_state_dict': self.eval_rnn.state_dict(),
            'qmix_state_dict': self.eval_qmix.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(dir_path, f"qmix_model_{step}.pth"))
        
        print("[*] Model saved")
        
    def load(self, dir_path, step=0):
        checkpoint = torch.load(os.path.join(dir_path, f"qmix_model_{step}.pth"))
        
        self.eval_rnn.load_state_dict(checkpoint['rnn_state_dict'])
        self.eval_qmix.load_state_dict(checkpoint['qmix_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix.load_state_dict(self.eval_qmix.state_dict())
        
        print("[*] Model loaded")