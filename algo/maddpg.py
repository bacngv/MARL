import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

from . import base

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
    def push(self, transition):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        batch_data = {}
        batch_data['obs'] = np.array([t['obs'] for t in batch])              # (B, num_agents, obs_dim)
        batch_data['feat'] = np.array([t['feat'] for t in batch])            # (B, num_agents, feat_dim)
        batch_data['acts'] = np.array([t['acts'] for t in batch])            # (B, num_agents, act_dim)
        batch_data['rewards'] = np.array([t['rewards'] for t in batch])        # (B, num_agents)
        batch_data['alives'] = np.array([t['alives'] for t in batch])          # (B, num_agents) 
        batch_data['global_state'] = np.array([t['global_state'] for t in batch])  # (B, global_state_dim)
        return batch_data
    def __len__(self):
        return len(self.buffer)

class MADDPG(base.ValueNet):
    def __init__(self, env, name, handle, num_agents, obs_dim, feat_dim, act_dim,
                 memory_size=2**10, batch_size=64, gamma=0.95, tau=0.01,
                 actor_lr=1e-3, critic_lr=1e-3):
        super().__init__(env, name, handle, update_every=1, learning_rate=actor_lr, gamma=gamma)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.feat_dim = feat_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.global_state_dim = num_agents * (obs_dim + feat_dim)
        self.replay_buffer = ReplayBuffer(memory_size)
        self.actors = []
        self.target_actors = []
        self.actor_opts = []
        self.critics = []
        self.target_critics = []
        self.critic_opts = []
        for i in range(num_agents):
            actor = self._construct_actor()
            target_actor = self._construct_actor()
            target_actor.load_state_dict(actor.state_dict())
            critic = self._construct_critic()
            target_critic = self._construct_critic()
            target_critic.load_state_dict(critic.state_dict())
            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.critics.append(critic)
            self.target_critics.append(target_critic)
            self.actor_opts.append(Adam(actor.parameters(), lr=actor_lr))
            self.critic_opts.append(Adam(critic.parameters(), lr=critic_lr))
    def _construct_actor(self):
        return nn.Sequential(
            nn.Linear(self.obs_dim + self.feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.act_dim),
            nn.Tanh()  
        )
    def _construct_critic(self):
        # Critic input: [global_state, joint_action]
        input_dim = self.global_state_dim + self.num_agents * self.act_dim
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def get_global_state(self, obs_all, feat_all):
        global_list = []
        for o, f in zip(obs_all, feat_all):
            o_flat = np.array(o).flatten()
            f_flat = np.array(f).flatten()
            global_list.append(np.concatenate([o_flat, f_flat]))
        global_state = np.concatenate(global_list)
        expected_length = self.global_state_dim
        current_length = global_state.shape[0]
        if current_length < expected_length:
            padding = np.zeros(expected_length - current_length)
            global_state = np.concatenate([global_state, padding])
        elif current_length > expected_length:
            global_state = global_state[:expected_length]
        return global_state
    def act(self, obs, feat, noise_scale=0.1):
        actions = []
        for i in range(self.num_agents):
            inp = np.concatenate([np.array(obs[i]).flatten(), np.array(feat[i]).flatten()])
            inp_tensor = torch.FloatTensor(inp).unsqueeze(0)  # (1, obs_dim+feat_dim)
            action = self.actors[i](inp_tensor).squeeze(0).detach().cpu().numpy()
            noise = noise_scale * np.random.randn(self.act_dim)
            action = action + noise
            action = np.clip(action, -1, 1)
            actions.append(action)
        return actions
    def flush_buffer(self, **kwargs):
        global_state = self.get_global_state(kwargs['state'][0], kwargs['state'][1])
        transition = {
            'obs': kwargs['state'][0],    
            'feat': kwargs['state'][1],   
            'acts': kwargs['acts'],       
            'rewards': kwargs['rewards'], 
            'alives': kwargs['alives'],   
            'global_state': global_state
        }
        self.replay_buffer.push(transition)
    def store_transition(self, obs, feat, acts, rewards, alives):
        global_state = self.get_global_state(obs, feat)
        transition = {
            'obs': obs,
            'feat': feat,
            'acts': acts,
            'rewards': rewards,
            'alives': alives,
            'global_state': global_state
        }
        self.replay_buffer.push(transition)
    def train(self, cuda=False):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        # convert to tensor
        # obs: (B, num_agents, obs_dim)
        obs_batch = torch.FloatTensor(batch['obs'])
        feat_batch = torch.FloatTensor(batch['feat'])
        acts_batch = torch.FloatTensor(batch['acts'])
        rewards_batch = torch.FloatTensor(batch['rewards'])
        alives_batch = torch.FloatTensor(batch['alives'])
        global_state_batch = torch.FloatTensor(batch['global_state'])
        device = next(self.actors[0].parameters()).device
        obs_batch = obs_batch.to(device)
        feat_batch = feat_batch.to(device)
        acts_batch = acts_batch.to(device)
        rewards_batch = rewards_batch.to(device)
        alives_batch = alives_batch.to(device)
        global_state_batch = global_state_batch.to(device)
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        # agent upate
        for agent_idx in range(self.num_agents):
            # --- Critic update ---
            with torch.no_grad():
                target_actions = []
                for j in range(self.num_agents):
                    inp_j = torch.cat([obs_batch[:, j, :], feat_batch[:, j, :]], dim=1)
                    target_action = self.target_actors[j](inp_j)
                    target_actions.append(target_action)
                # (B, num_agents*act_dim)
                target_actions = torch.cat(target_actions, dim=1)
                target_input = torch.cat([global_state_batch, target_actions], dim=1)
                target_Q = self.target_critics[agent_idx](target_input).squeeze()  # (B,)
                reward = rewards_batch[:, agent_idx]
                done = 1 - alives_batch[:, agent_idx]
                y = reward + self.gamma * target_Q * (1 - done)
            current_actions = acts_batch.view(self.batch_size, -1)  # (B, num_agents*act_dim)
            critic_input = torch.cat([global_state_batch, current_actions], dim=1)
            current_Q = self.critics[agent_idx](critic_input).squeeze()  # (B,)
            critic_loss = F.mse_loss(current_Q, y.detach())
            self.critic_opts[agent_idx].zero_grad()
            critic_loss.backward()
            self.critic_opts[agent_idx].step()
            total_critic_loss += critic_loss.item()
            # --- Actor update ---
            current_actions_list = []
            for j in range(self.num_agents):
                if j == agent_idx:
                    inp = torch.cat([obs_batch[:, j, :], feat_batch[:, j, :]], dim=1)
                    current_action = self.actors[j](inp)
                else:
                    current_action = acts_batch[:, j, :]
                current_actions_list.append(current_action)
            current_actions_concat = torch.cat(current_actions_list, dim=1)  # (B, num_agents*act_dim)
            actor_input = torch.cat([global_state_batch, current_actions_concat], dim=1)
            actor_loss = -self.critics[agent_idx](actor_input).mean()

            self.actor_opts[agent_idx].zero_grad()
            actor_loss.backward()
            self.actor_opts[agent_idx].step()

            total_actor_loss += actor_loss.item()
            # --- Soft update cho target networks ---
            for target_param, param in zip(self.target_critics[agent_idx].parameters(),
                                           self.critics[agent_idx].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_actors[agent_idx].parameters(),
                                           self.actors[agent_idx].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        print(f"[MADDPG] Actor Loss: {total_actor_loss/self.num_agents:.4f}, "
              f"Critic Loss: {total_critic_loss/self.num_agents:.4f}")
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        for i in range(self.num_agents):
            actor_path = os.path.join(dir_path, f"maddpg_actor_agent_{i}_{step}.pt")
            critic_path = os.path.join(dir_path, f"maddpg_critic_agent_{i}_{step}.pt")
            torch.save(self.actors[i].state_dict(), actor_path)
            torch.save(self.critics[i].state_dict(), critic_path)
        print("[*] MADDPG models saved")
    def load(self, dir_path, step=0):
        for i in range(self.num_agents):
            actor_path = os.path.join(dir_path, f"maddpg_actor_agent_{i}_{step}.pt")
            critic_path = os.path.join(dir_path, f"maddpg_critic_agent_{i}_{step}.pt")
            self.actors[i].load_state_dict(torch.load(actor_path))
            self.critics[i].load_state_dict(torch.load(critic_path))
        print("[*] MADDPG models loaded")
