import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from . import base
from .. import replaybuffer

class PPO(base.ValueNet):
    def __init__(self, env, name, handle, sub_len, memory_size=2**10, batch_size=64,
                 update_every=5, use_mf=False, learning_rate=0.0001, clip_param=0.3,
                 value_coef=0.5, entropy_coef=0.01, gamma=0.95, gae_lambda=0.95, tau=0.01):
        super().__init__(env, name, handle, update_every=update_every, use_mf=use_mf,
                        learning_rate=learning_rate, gamma=gamma)
        
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.tau = tau
        
        self.replay_buffer = replaybuffer.SimpleMemoryGroup(self.view_space, self.feature_space,
                                             self.num_actions, memory_size, batch_size, sub_len)
        self.actor = self._construct_actor()
        self.critic = self._construct_critic()
        self.target_actor = self._construct_actor()
        self.target_critic = self._construct_critic()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optim = torch.optim.Adam(self.get_params(self.actor), lr=learning_rate)
        self.critic_optim = torch.optim.Adam(self.get_params(self.critic), lr=learning_rate)
    
    def get_all_params(self):
        """
        Get all parameters from both actor and critic networks for self-play.
        
        Returns:
            list: List of all parameters from both networks
        """
        params = []
        for k, v in self.actor.items():
            params.extend(list(v.parameters()))
        for k, v in self.critic.items():
            params.extend(list(v.parameters()))
        return params
    def get_params(self, network_dict):
        """Helper method to get parameters from a network dictionary"""
        params = []
        for k, v in network_dict.items():
            params.extend(list(v.parameters()))
        return params
    def update(self):
        """Soft update target networks using tau parameter"""
        # actor
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)  
        # critic
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def flush_buffer(self, **kwargs):
        """
        Push data to the replay buffer.
        
        Args:
            **kwargs: Dictionary containing the following keys:
                - obs: observation
                - feature: additional features
                - acts: actions taken
                - rewards: rewards received
                - alives: alive status
                - obs_next: next observation
                - feature_next: next additional features
        """
        self.replay_buffer.push(**kwargs)
    def _construct_actor(self):
        temp_dict = nn.ModuleDict()
        temp_dict['conv1'] = nn.Conv2d(in_channels=self.view_space[2], out_channels=32, kernel_size=3)
        temp_dict['conv2'] = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        temp_dict['obs_linear'] = nn.Linear(self.get_flatten_dim(temp_dict), 256)
        temp_dict['emb_linear'] = nn.Linear(self.feature_space, 32)
        if self.use_mf:
            temp_dict['prob_emb_linear'] = nn.Sequential(
                nn.Linear(self.num_actions, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
        temp_dict['final_linear'] = nn.Sequential(
            nn.Linear(320 if self.use_mf else 288, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )
        return temp_dict
    def _construct_critic(self):
        temp_dict = nn.ModuleDict()
        temp_dict['conv1'] = nn.Conv2d(in_channels=self.view_space[2], out_channels=32, kernel_size=3)
        temp_dict['conv2'] = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        temp_dict['obs_linear'] = nn.Linear(self.get_flatten_dim(temp_dict), 256)
        temp_dict['emb_linear'] = nn.Linear(self.feature_space, 32)
        if self.use_mf:
            temp_dict['prob_emb_linear'] = nn.Sequential(
                nn.Linear(self.num_actions, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
        temp_dict['final_linear'] = nn.Sequential(
            nn.Linear(320 if self.use_mf else 288, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        return temp_dict
    def get_value(self, obs, feature, prob=None):
        c_h = F.relu(self.critic['conv2'](F.relu(self.critic['conv1'](obs)))).flatten(start_dim=1)
        c_h = torch.cat([self.critic['obs_linear'](c_h), self.critic['emb_linear'](feature)], -1)
        if self.use_mf:
            c_h = torch.cat([c_h, self.critic['prob_emb_linear'](prob)], -1)
        value = self.critic['final_linear'](c_h)
        return value.squeeze(-1)
    def get_action_and_value(self, obs, feature, prob=None, action=None):
        a_h = F.relu(self.actor['conv2'](F.relu(self.actor['conv1'](obs)))).flatten(start_dim=1)
        a_h = torch.cat([self.actor['obs_linear'](a_h), self.actor['emb_linear'](feature)], -1)
        if self.use_mf:
            a_h = torch.cat([a_h, self.actor['prob_emb_linear'](prob)], -1)
        logits = self.actor['final_linear'](a_h)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        if action is None:
            action = dist.sample()
        value = self.get_value(obs, feature, prob)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value
    def act(self, obs, feature, prob=None, eps=None):
        with torch.no_grad():
            action, _, _, _ = self.get_action_and_value(obs, feature, prob)
        return action.cpu().numpy()
    def train(self, cuda):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()
        total_loss = 0
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for i in range(batch_num):
            obs, feat, obs_next, feat_next, dones, rewards, acts, masks = self.replay_buffer.sample()
            obs = torch.FloatTensor(obs).permute([0, 3, 1, 2])
            obs_next = torch.FloatTensor(obs_next).permute([0, 3, 1, 2])
            feat = torch.FloatTensor(feat)
            feat_next = torch.FloatTensor(feat_next)
            acts = torch.LongTensor(acts)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)
            masks = torch.FloatTensor(masks)
            if cuda:
                obs = obs.cuda()
                obs_next = obs_next.cuda()
                feat = feat.cuda()
                feat_next = feat_next.cuda()
                acts = acts.cuda()
                rewards = rewards.cuda()
                dones = dones.cuda()
                masks = masks.cuda()
            _, new_log_prob, entropy, values = self.get_action_and_value(obs, feat, action=acts)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            with torch.no_grad():
                next_values = self.get_value(obs_next, feat_next)
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = next_values[t]
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # PPO policy loss
            ratio = torch.exp(new_log_prob)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            # Value loss
            value_loss = F.mse_loss(values, returns)
            # Entropy loss
            entropy_loss = -entropy.mean()
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            # Optimize
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()
            
            total_loss += loss.item()
            actor_loss += policy_loss.item()
            critic_loss += value_loss.item()
            entropy_loss += entropy_loss.item()
            
            if i % 50 == 0:
                print(f'[*] LOSS: {loss.item():.4f} (Policy: {policy_loss.item():.4f}, '
                      f'Value: {value_loss.item():.4f}, Entropy: {entropy_loss.item():.4f})')
                
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        actor_path = os.path.join(dir_path, f"ppo_actor_{step}")
        critic_path = os.path.join(dir_path, f"ppo_critic_{step}")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print("[*] Model saved")
        
    def load(self, dir_path, step=0):
        actor_path = os.path.join(dir_path, f"ppo_actor_{step}")
        critic_path = os.path.join(dir_path, f"ppo_critic_{step}")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        print("[*] Loaded model")