import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from . import base


class MetaBuffer(object):
    def __init__(self, shape, max_len, dtype='float32'):
        self.max_len = max_len
        self.data = np.zeros([max_len] + list(shape if isinstance(shape, tuple) else [shape])).astype(dtype)
        self.start = 0
        self.length = 0
        self._flag = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[idx]

    def sample(self, idx):
        return self.data[idx % self.length]

    def pull(self):
        return self.data[:self.length]

    def append(self, value):
        start = 0
        num = len(value)

        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag:] = value[:tail]
            num -= tail
            start = tail
            self._flag = 0

        self.data[self._flag:self._flag + num] = value[start:]
        self._flag += num
        self.length = min(self.length + len(value), self.max_len)

    def reset_new(self, start, value):
        self.data[start:] = value


class AgentMemory(object):
    def __init__(self, obs_shape, feat_shape, act_n, max_len, use_mean=False, global_state_shape=None):
        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.use_mean = use_mean
        if self.use_mean:
            self.prob = MetaBuffer((act_n,), max_len)
        self.global_state = MetaBuffer(global_state_shape, max_len)

    def append(self, obs0, feat0, act, reward, alive, prob=None, global_state=None):
        self.obs0.append(np.array([obs0]))
        self.feat0.append(np.array([feat0]))
        self.actions.append(np.array([act], dtype=np.int32))
        self.rewards.append(np.array([reward]))
        self.terminals.append(np.array([not alive], dtype=bool))
        if self.use_mean:
            self.prob.append(np.array([prob]))
        self.global_state.append(np.array([global_state]))


    def pull(self):
        res = {
            'obs0': self.obs0.pull(),
            'feat0': self.feat0.pull(),
            'act': self.actions.pull(),
            'rewards': self.rewards.pull(),
            'terminals': self.terminals.pull(),
            'prob': None if not self.use_mean else self.prob.pull(),
            'global_state': self.global_state.pull()
        }
        return res



class MemoryGroup(object):
    def __init__(self, obs_shape, feat_shape, act_n, max_len, batch_size, sub_len, use_mean=False):
        self.agent = dict()
        self.max_len = max_len
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.feat_shape = feat_shape
        self.sub_len = sub_len
        self.use_mean = use_mean
        self.act_n = act_n

        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.masks = MetaBuffer((), max_len, dtype='bool')
        if use_mean:
            self.prob = MetaBuffer((act_n,), max_len)
        self._new_add = 0

    def _flush(self, **kwargs):
        self.obs0.append(kwargs['obs0'])
        self.feat0.append(kwargs['feat0'])
        self.actions.append(kwargs['act'])
        self.rewards.append(kwargs['rewards'])
        self.terminals.append(kwargs['terminals'])
        if self.use_mean:
            self.prob.append(kwargs['prob'])
        mask = np.where(kwargs['terminals'] == True, False, True)
        mask[-1] = False
        self.masks.append(mask)

    def push(self, **kwargs):
        for i, _id in enumerate(kwargs['ids']):
            if self.agent.get(_id) is None:
                self.agent[_id] = AgentMemory(self.obs_shape, self.feat_shape, self.act_n, self.sub_len, use_mean=self.use_mean)
            if self.use_mean:
                self.agent[_id].append(
                    obs0=kwargs['state'][0][i],
                    feat0=kwargs['state'][1][i],
                    act=kwargs['acts'][i],
                    reward=kwargs['rewards'][i],
                    alive=kwargs['alives'][i],
                    prob=kwargs['prob'][i]
                )
            else:
                self.agent[_id].append(
                    obs0=kwargs['state'][0][i],
                    feat0=kwargs['state'][1][i],
                    act=kwargs['acts'][i],
                    reward=kwargs['rewards'][i],
                    alive=kwargs['alives'][i]
                )

    def tight(self):
        ids = list(self.agent.keys())
        np.random.shuffle(ids)
        for ele in ids:
            tmp = self.agent[ele].pull()
            self._new_add += len(tmp['obs0'])
            self._flush(**tmp)
        self.agent = dict()  # clear

    def sample(self):
        idx = np.random.choice(self.nb_entries, size=self.batch_size)
        next_idx = (idx + 1) % self.nb_entries

        obs = self.obs0.sample(idx)
        obs_next = self.obs0.sample(next_idx)
        feature = self.feat0.sample(idx)
        feature_next = self.feat0.sample(next_idx)
        actions = self.actions.sample(idx)
        rewards = self.rewards.sample(idx)
        dones = self.terminals.sample(idx)
        masks = self.masks.sample(idx)

        if self.use_mean:
            act_prob = self.prob.sample(idx)
            act_next_prob = self.prob.sample(next_idx)
        else:
            act_prob = None
            act_next_prob = None
        return obs, feature, obs_next, feature_next, dones, rewards, actions, masks

    def get_batch_num(self):
        print('\n[INFO] Length of buffer and new add:', len(self.obs0), self._new_add)
        res = self._new_add * 2 // self.batch_size
        self._new_add = 0
        return res

    @property
    def nb_entries(self):
        return len(self.obs0)

class MAMemoryGroup(MemoryGroup):
    def __init__(self, obs_shape, feat_shape, global_state_shape, act_n, max_len, batch_size, sub_len, use_mean=False):
        super().__init__(obs_shape, feat_shape, act_n, max_len, batch_size, sub_len, use_mean)
        self.global_states = MetaBuffer(global_state_shape, max_len)

    def _flush(self, **kwargs):
        super()._flush(**kwargs)
        self.global_states.append(kwargs['global_state'])

    def push(self, **kwargs):
        n_agents = len(kwargs['ids'])
        global_states_array = np.array([kwargs['global_state']] * n_agents)
        kwargs['global_state'] = global_states_array
        for i, _id in enumerate(kwargs['ids']):
            if self.agent.get(_id) is None:
                self.agent[_id] = AgentMemory(self.obs_shape, self.feat_shape, self.act_n, self.sub_len, use_mean=self.use_mean, global_state_shape=self.global_states.data.shape[1:])
            if self.use_mean:
                self.agent[_id].append(
                    obs0=kwargs['state'][0][i],
                    feat0=kwargs['state'][1][i],
                    act=kwargs['acts'][i],
                    reward=kwargs['rewards'][i],
                    alive=kwargs['alives'][i],
                    prob=kwargs['prob'][i],
                    global_state=global_states_array[i]
                )
            else:
                self.agent[_id].append(
                    obs0=kwargs['state'][0][i],
                    feat0=kwargs['state'][1][i],
                    act=kwargs['acts'][i],
                    reward=kwargs['rewards'][i],
                    alive=kwargs['alives'][i],
                    global_state=global_states_array[i]
                )

    def sample(self):
        idx = np.random.choice(self.nb_entries, size=self.batch_size)
        next_idx = (idx + 1) % self.nb_entries

        obs = self.obs0.sample(idx)
        obs_next = self.obs0.sample(next_idx)
        feat = self.feat0.sample(idx)
        feat_next = self.feat0.sample(next_idx)
        actions = self.actions.sample(idx)
        rewards = self.rewards.sample(idx)
        dones = self.terminals.sample(idx)
        masks = self.masks.sample(idx)

        if self.use_mean:
            act_prob = self.prob.sample(idx)
            act_next_prob = self.prob.sample(next_idx)
        else:
            act_prob = None
            act_next_prob = None

        global_state = self.global_states.sample(idx)
        global_state_next = self.global_states.sample(next_idx)
        return obs, feat, obs_next, feat_next, dones, rewards, actions, masks, global_state, global_state_next

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
        self.replay_buffer = MAMemoryGroup(self.view_space, self.feature_space,
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
    def get_value(self, global_state):
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
        value = None  # global state ở critic
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
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
            obs, feat, obs_next, feat_next, dones, rewards, acts, masks, global_state, global_state_next = self.replay_buffer.sample()

            obs = torch.FloatTensor(obs).permute([0, 3, 1, 2])
            obs_next = torch.FloatTensor(obs_next).permute([0, 3, 1, 2])
            feat = torch.FloatTensor(feat)
            feat_next = torch.FloatTensor(feat_next)
            acts = torch.LongTensor(acts)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)
            masks = torch.FloatTensor(masks)
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
                global_state = global_state.cuda()
                global_state_next = global_state_next.cuda()
            _, new_log_prob, entropy, _ = self.get_action_and_value(obs, feat, action=acts)
            values = self.get_value(global_state)
            with torch.no_grad():
                next_values = self.get_value(global_state_next)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = next_values[t]
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ratio = torch.exp(new_log_prob)
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
