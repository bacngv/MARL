import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical
from . import base 


# --- Replay Buffer ---
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
        num = len(value)
        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag:] = value[:tail]
            num -= tail
            self.data[:num] = value[tail:]
            self._flag = num
        else:
            self.data[self._flag:self._flag + num] = value
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
        self.old_log_prob = MetaBuffer((), max_len)
        self.global_state = MetaBuffer(global_state_shape, max_len)

    def append(self, obs0, feat0, act, reward, alive, old_log_prob, prob=None, global_state=None):
        self.obs0.append(np.array([obs0]))
        self.feat0.append(np.array([feat0]))
        self.actions.append(np.array([act], dtype=np.int32))
        self.rewards.append(np.array([reward]))
        self.terminals.append(np.array([not alive], dtype=bool))
        self.old_log_prob.append(np.array([old_log_prob]))
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
            'old_log_prob': self.old_log_prob.pull(),
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
        self.old_log_prob = MetaBuffer((), max_len)
        if use_mean:
            self.prob = MetaBuffer((act_n,), max_len)
        self._new_add = 0

    def _flush(self, **kwargs):
        self.obs0.append(kwargs['obs0'])
        self.feat0.append(kwargs['feat0'])
        self.actions.append(kwargs['act'])
        self.rewards.append(kwargs['rewards'])
        self.terminals.append(kwargs['terminals'])
        self.old_log_prob.append(kwargs['old_log_prob'])
        if self.use_mean:
            self.prob.append(kwargs['prob'])
        mask = np.where(kwargs['terminals'] == True, False, True)
        if len(mask) > 0:
            mask[-1] = False
        self.masks.append(mask)

    def push(self, **kwargs):
        for i, _id in enumerate(kwargs['ids']):
            if _id not in self.agent:
                self.agent[_id] = AgentMemory(self.obs_shape, self.feat_shape, self.act_n, self.sub_len,
                                              use_mean=self.use_mean,
                                              global_state_shape=kwargs['global_state'][0].shape)
            if self.use_mean:
                self.agent[_id].append(
                    obs0=kwargs['state'][0][i],
                    feat0=kwargs['state'][1][i],
                    act=kwargs['acts'][i],
                    reward=kwargs['rewards'][i],
                    alive=kwargs['alives'][i],
                    old_log_prob=kwargs['old_log_prob'][i],
                    prob=kwargs['prob'][i],
                    global_state=kwargs['global_state'][i]
                )
            else:
                self.agent[_id].append(
                    obs0=kwargs['state'][0][i],
                    feat0=kwargs['state'][1][i],
                    act=kwargs['acts'][i],
                    reward=kwargs['rewards'][i],
                    alive=kwargs['alives'][i],
                    old_log_prob=kwargs['old_log_prob'][i],
                    global_state=kwargs['global_state'][i]
                )

    def tight(self):
        ids = list(self.agent.keys())
        np.random.shuffle(ids)
        for ele in ids:
            tmp = self.agent[ele].pull()
            self._new_add += len(tmp['obs0'])
            self._flush(**tmp)
        self.agent = dict()  # clear agent memories

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
        old_log_prob = self.old_log_prob.sample(idx)

        if self.use_mean:
            act_prob = self.prob.sample(idx)
            act_next_prob = self.prob.sample(next_idx)
        else:
            act_prob = None
            act_next_prob = None
        return obs, feature, obs_next, feature_next, dones, rewards, actions, masks, old_log_prob

    def get_batch_num(self):
        buffer_length = len(self.obs0)
        print('\n[INFO] Buffer length:', buffer_length, " new additions:", self._new_add)
        if buffer_length >= self.batch_size:
            res = buffer_length // self.batch_size
        else:
            res = 0
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
            if _id not in self.agent:
                self.agent[_id] = AgentMemory(self.obs_shape, self.feat_shape, self.act_n, self.sub_len,
                                              use_mean=self.use_mean,
                                              global_state_shape=self.global_states.data.shape[1:])
            if self.use_mean:
                self.agent[_id].append(
                    obs0=kwargs['state'][0][i],
                    feat0=kwargs['state'][1][i],
                    act=kwargs['acts'][i],
                    reward=kwargs['rewards'][i],
                    alive=kwargs['alives'][i],
                    old_log_prob=kwargs['old_log_prob'][i],
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
                    old_log_prob=kwargs['old_log_prob'][i],
                    global_state=global_states_array[i]
                )
    def sample(self):
        (obs, feat, obs_next, feat_next, dones, rewards,
         acts, masks, old_log_prob) = super().sample()
        indices = np.arange(len(self.obs0))[:self.batch_size]
        global_state = self.global_states.sample(indices)
        global_state_next = self.global_states.sample((indices + 1) % len(self.obs0))
        return obs, feat, obs_next, feat_next, dones, rewards, acts, masks, old_log_prob, global_state, global_state_next
# --- Value Decomposition Network) ---
class VDN(base.ValueNet):
    def __init__(self, env, name, handle, sub_len, num_agents, memory_size=2**10, batch_size=64,
                 update_every=5, learning_rate=0.0001, gamma=0.95, tau=0.01, use_mf=False):
        super().__init__(env, name, handle, update_every=update_every, use_mf=use_mf,
                         learning_rate=learning_rate, gamma=gamma)
        self.num_agents = num_agents
        local_obs_dim = np.prod(self.view_space)  # H x W x C
        local_feat_dim = self.feature_space
        global_state_dim = num_agents * (local_obs_dim + local_feat_dim)
        self.global_state_dim = global_state_dim
        self.gamma = gamma
        self.tau = tau
        self.use_mf = use_mf
        self.replay_buffer = MAMemoryGroup(self.view_space, self.feature_space,
                                           (global_state_dim,), self.num_actions,
                                           memory_size, batch_size, sub_len, use_mean=use_mf)
        self.q_net = self._construct_q_net()
        self.target_q_net = self._construct_q_net()
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
    def _construct_q_net(self):
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
        final_in_dim = 320 if self.use_mf else 288
        temp_dict['final_linear'] = nn.Sequential(
            nn.Linear(final_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )
        return temp_dict
    def get_all_params(self):
        return list(self.q_net.parameters())
    def get_params(self, network):
        return list(network.parameters())
    def get_action_and_value(self, obs, feature, prob=None, action=None):
        x = F.relu(self.q_net['conv1'](obs))
        x = F.relu(self.q_net['conv2'](x))
        x = x.flatten(start_dim=1)
        x = self.q_net['obs_linear'](x)
        y = self.q_net['emb_linear'](feature)
        combined = torch.cat([x, y], dim=-1)
        if self.use_mf and prob is not None:
            combined = torch.cat([combined, self.q_net['prob_emb_linear'](prob)], dim=-1)
        q_values = self.q_net['final_linear'](combined)  # (batch, num_actions)
        if action is None:
            action = q_values.argmax(dim=-1)
        # Dummy log_prob and entropy
        dummy_log_prob = torch.zeros(q_values.shape[0]).to(q_values.device)
        dummy_entropy = torch.zeros(q_values.shape[0]).to(q_values.device)
        # Q-value
        chosen_q = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action, dummy_log_prob, dummy_entropy, chosen_q
    def act(self, obs, feature, prob=None, eps=None):
        with torch.no_grad():
            action, _, _, _ = self.get_action_and_value(obs, feature, prob)
        return action.cpu().numpy()
    def update_target_network(self):
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def flush_buffer(self, **kwargs):
        global_state = self.get_global_state(kwargs['state'][0], kwargs['state'][1])
        kwargs['global_state'] = global_state
        device = next(self.q_net.parameters()).device
        obs_tensor = torch.FloatTensor(np.array(kwargs['state'][0])).permute(0, 3, 1, 2).to(device)
        feat_tensor = torch.FloatTensor(np.array(kwargs['state'][1])).to(device)
        if kwargs.get('prob') is not None:
            prob_tensor = torch.FloatTensor(np.array(kwargs['prob'])).to(device)
        else:
            prob_tensor = None
        acts_tensor = torch.LongTensor(kwargs['acts']).to(device)
        # dummy zeros
        with torch.no_grad():
            _, _, _, _ = self.get_action_and_value(obs_tensor, feat_tensor, prob_tensor, action=acts_tensor)
        kwargs['old_log_prob'] = np.zeros_like(acts_tensor.cpu().numpy(), dtype=np.float32)
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
    def train(self, cuda=False):
        """
          loss = MSE( Q(s,a) - (r + γ * maxₐ' Q(s',a') ) )
        """
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()
        if batch_num == 0:
            print("[VDN] Not enough samples to train")
            return
        total_loss = 0
        for i in range(batch_num):
            sample_data = self.replay_buffer.sample()
            (obs, feat, obs_next, feat_next, dones, rewards,
             acts, masks, old_log_prob, global_state, global_state_next) = sample_data
            obs = torch.FloatTensor(obs).permute([0, 3, 1, 2])
            obs_next = torch.FloatTensor(obs_next).permute([0, 3, 1, 2])
            feat = torch.FloatTensor(feat)
            feat_next = torch.FloatTensor(feat_next)
            acts = torch.LongTensor(acts)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)
            if cuda:
                obs = obs.cuda()
                obs_next = obs_next.cuda()
                feat = feat.cuda()
                feat_next = feat_next.cuda()
                acts = acts.cuda()
                rewards = rewards.cuda()
                dones = dones.cuda()
            # Q(s,a)
            _, _, _, current_q = self.get_action_and_value(obs, feat, action=acts)
            # target: r + γ * maxₐ' Q_target(s',a')
            x_next = F.relu(self.target_q_net['conv1'](obs_next))
            x_next = F.relu(self.target_q_net['conv2'](x_next))
            x_next = x_next.flatten(start_dim=1)
            x_next = self.target_q_net['obs_linear'](x_next)
            y_next = self.target_q_net['emb_linear'](feat_next)
            combined_next = torch.cat([x_next, y_next], dim=-1)
            if self.use_mf:
                pass
            q_next = self.target_q_net['final_linear'](combined_next)
            max_q_next, _ = q_next.max(dim=-1)
            target_q = rewards + self.gamma * max_q_next * (1 - dones)
            loss = F.mse_loss(current_q, target_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.update_target_network()
            if i % 50 == 0:
                print(f'[VDN] Batch {i}/{batch_num} Loss: {loss.item():.4f}')
        print(f"[VDN] Total loss: {total_loss:.4f}")
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        qnet_path = os.path.join(dir_path, f"vdn_qnet_{step}")
        torch.save(self.q_net.state_dict(), qnet_path)
        print("[*] VDN model saved")
    def load(self, dir_path, step=0):
        qnet_path = os.path.join(dir_path, f"vdn_qnet_{step}")
        self.q_net.load_state_dict(torch.load(qnet_path))
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        print("[*] VDN model loaded")
