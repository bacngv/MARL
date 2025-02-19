import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from . import base
from .. import replaybuffer

# CTDE MAPPO
class MAPPO(base.ValueNet):
    def __init__(self, env, name, handle, sub_len, num_agents, memory_size=2**10, batch_size=64,
                 update_every=5, use_mf=False, learning_rate=0.0001, clip_param=0.3,
                 value_coef=0.5, entropy_coef=0.01, gamma=0.95, gae_lambda=0.95, tau=0.01):
        super().__init__(env, name, handle, update_every=update_every, use_mf=use_mf,
                         learning_rate=learning_rate, gamma=gamma)
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.tau = tau
        self.num_agents = num_agents
        local_obs_dim = np.prod(self.view_space)  # (H x W x C)
        local_feat_dim = self.feature_space
        global_state_dim = num_agents * (local_obs_dim + local_feat_dim)
        self.global_state_dim = global_state_dim
        self.replay_buffer = replaybuffer.MAMemoryGroup(self.view_space, self.feature_space,
                                           (global_state_dim,), self.num_actions,
                                           memory_size, batch_size, sub_len, use_mean=use_mf)
        self.actor = self._construct_actor()
        self.critic = self._construct_critic()
        self.target_actor = self._construct_actor()
        self.target_critic = self._construct_critic()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optim = torch.optim.Adam(self.get_params(self.actor), lr=learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
    def get_all_params(self):
        params = []
        for k, v in self.actor.items():
            params.extend(list(v.parameters()))
        for param in self.critic.parameters():
            params.append(param)
        return params
    def get_params(self, network_dict):
        params = []
        for k, v in network_dict.items():
            params.extend(list(v.parameters()))
        return params
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
        model = nn.Sequential(
            nn.Linear(self.global_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        return model
    def get_value(self, global_state, use_target=False):
        if use_target:
            value = self.target_critic(global_state)
        else:
            value = self.critic(global_state)
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
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = None
        return action, log_prob, entropy, value
    def act(self, obs, feature, prob=None, eps=None):
        with torch.no_grad():
            action, _, _, _ = self.get_action_and_value(obs, feature, prob)
        return action.cpu().numpy()
    def update(self):
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def flush_buffer(self, **kwargs):
        global_state = self.get_global_state(kwargs['state'][0], kwargs['state'][1])
        kwargs['global_state'] = global_state
        device = next(self.actor.parameters()).device
        obs_tensor = torch.FloatTensor(np.array(kwargs['state'][0])).permute(0, 3, 1, 2).to(device)
        feat_tensor = torch.FloatTensor(np.array(kwargs['state'][1])).to(device)
        if kwargs.get('prob') is not None:
            prob_tensor = torch.FloatTensor(np.array(kwargs['prob'])).to(device)
        else:
            prob_tensor = None
        acts_tensor = torch.LongTensor(kwargs['acts']).to(device)

        with torch.no_grad():
            _, log_prob, _, _ = self.get_action_and_value(obs_tensor, feat_tensor, prob_tensor, action=acts_tensor)
        kwargs['old_log_prob'] = log_prob.cpu().numpy()
        self.replay_buffer.push(**kwargs)
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
    def train(self, cuda):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()
        total_loss = 0
        for i in range(batch_num):
            sample_data = self.replay_buffer.sample()
            obs, feat, obs_next, feat_next, dones, rewards, acts, masks, old_log_prob, global_state, global_state_next = sample_data
            obs = torch.FloatTensor(obs).permute([0, 3, 1, 2])
            obs_next = torch.FloatTensor(obs_next).permute([0, 3, 1, 2])
            feat = torch.FloatTensor(feat)
            feat_next = torch.FloatTensor(feat_next)
            acts = torch.LongTensor(acts)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)
            masks = torch.FloatTensor(masks)
            old_log_prob = torch.FloatTensor(old_log_prob)
            global_state = torch.FloatTensor(global_state)
            global_state_next = torch.FloatTensor(global_state_next)
            if cuda:
                obs = obs.cuda()
                obs_next = obs_next.cuda()
                feat = feat.cuda()
                feat_next = feat_next.cuda()
                acts = acts.cuda()
                rewards = rewards.cuda()
                dones = dones.cuda()
                masks = masks.cuda()
                old_log_prob = old_log_prob.cuda()
                global_state = global_state.cuda()
                global_state_next = global_state_next.cuda()
            _, new_log_prob, entropy, _ = self.get_action_and_value(obs, feat, action=acts)
            # r = exp(new_log_prob - old_log_prob)
            ratio = torch.exp(new_log_prob - old_log_prob)
            values = self.get_value(global_state, use_target=False)
            with torch.no_grad():
                next_values = self.get_value(global_state_next, use_target=True)
            # gae
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalue = next_values[t]
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalue = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalue * nextnonterminal - values[t]
                lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropy.mean()
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()
            total_loss += loss.item()
            if i % 50 == 0:
                print(f'[MAPPO] LOSS: {loss.item():.4f} (Policy: {policy_loss.item():.4f}, '
                      f'Value: {value_loss.item():.4f}, Entropy: {entropy_loss.item():.4f})')
        print(f"[MAPPO] Total loss: {total_loss:.4f}")
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        actor_path = os.path.join(dir_path, f"mappo_actor_{step}")
        critic_path = os.path.join(dir_path, f"mappo_critic_{step}")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print("[*] Model saved")
    def load(self, dir_path, step=0):
        actor_path = os.path.join(dir_path, f"mappo_actor_{step}")
        critic_path = os.path.join(dir_path, f"mappo_critic_{step}")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        print("[*] Loaded model")