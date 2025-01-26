import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class RNNAgent(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim, output_dim):
        super(RNNAgent, self).__init__()
        
        # Feature extraction layers
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        
        # Recurrent layer
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        
        # Output layer
        self.fc2 = nn.Linear(rnn_hidden_dim, output_dim)
        
    def forward(self, obs, hidden_state):
        # Feature extraction
        x = F.relu(self.fc1(obs))
        
        # Update hidden state
        h_next = self.rnn(x, hidden_state)
        
        # Q-value output
        q_values = self.fc2(h_next)
        
        return q_values, h_next

class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_embed_dim=32):
        super(QMixer, self).__init__()
        
        # Hypernetwork for mixing
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, n_agents)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
        
        self.hyper_b1 = nn.Linear(state_dim, 1)
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, individual_qs, states):
        # First layer mixing
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        
        hidden = F.elu(torch.bmm(individual_qs.unsqueeze(2), w1.unsqueeze(1)).squeeze(2) + b1)
        
        # Second layer mixing
        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        
        q_tot = torch.bmm(hidden.unsqueeze(1), w2.unsqueeze(2)).squeeze(2) + b2
        
        return q_tot.squeeze(1)

class QMIX:
    def __init__(self, config):
        self.n_agents = config['n_agents']
        self.input_shape = config['input_shape']
        self.rnn_hidden_dim = config['rnn_hidden_dim']
        self.output_dim = config['output_dim']
        self.lr = config.get('lr', 1e-4)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.01)
        
        # Agent networks
        self.agents = nn.ModuleList([
            RNNAgent(self.input_shape, self.rnn_hidden_dim, self.output_dim) 
            for _ in range(self.n_agents)
        ])
        self.target_agents = nn.ModuleList([
            RNNAgent(self.input_shape, self.rnn_hidden_dim, self.output_dim)
            for _ in range(self.n_agents)
        ])
        
        # Mixer networks
        self.mixer = QMixer(self.n_agents, config['state_dim'])
        self.target_mixer = QMixer(self.n_agents, config['state_dim'])
        
        # Sync target networks
        self.target_agents.load_state_dict(self.agents.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.agents.parameters()) + list(self.mixer.parameters()), 
            lr=self.lr
        )
        
    def select_actions(self, obs, hidden_states, epsilon=0):
        actions = []
        next_hidden_states = []
        
        for agent, obs_agent, h_state in zip(self.agents, obs, hidden_states):
            # Add exploration
            if np.random.random() < epsilon:
                action = np.random.randint(self.output_dim)
            else:
                with torch.no_grad():
                    q_values, h_next = agent(torch.FloatTensor(obs_agent), h_state)
                    action = q_values.argmax().item()
            
            actions.append(action)
            next_hidden_states.append(h_next)
        
        return actions, next_hidden_states
    
    def train(self, batch):
        # Unpack batch
        obs, actions, rewards, next_obs, done, state = batch
        
        # Compute individual Q-values
        q_values = []
        target_q_values = []
        
        for agent, obs_agent in zip(self.agents, obs):
            q, _ = agent(obs_agent)
            q_values.append(q)
        
        for target_agent, next_obs_agent in zip(self.target_agents, next_obs):
            q, _ = target_agent(next_obs_agent)
            target_q_values.append(q)
        
        # Stack Q-values
        q_values = torch.stack(q_values, dim=1)
        target_q_values = torch.stack(target_q_values, dim=1)
        
        # Compute centralized Q-values with mixer
        q_tot = self.mixer(q_values, state)
        target_q_tot = self.target_mixer(target_q_values, state)
        
        # Compute targets
        targets = rewards + self.gamma * target_q_tot * (1 - done)
        
        # Compute loss
        loss = F.mse_loss(q_tot, targets.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
        self.optimizer.step()
        
        # Soft update target networks
        self._soft_update()
        
        return loss.item()
    
    def _soft_update(self):
        for agent, target_agent in zip(self.agents, self.target_agents):
            for param, target_param in zip(agent.parameters(), target_agent.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        
        for param, target_param in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path):
        torch.save({
            'agents': self.agents.state_dict(),
            'mixer': self.mixer.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.agents.load_state_dict(checkpoint['agents'])
        self.mixer.load_state_dict(checkpoint['mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])