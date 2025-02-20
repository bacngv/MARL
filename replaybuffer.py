import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from moviepy.editor import ImageSequenceClip
import threading
import pandas as pd

class Color:
    INFO = '\033[1;34m{}\033[0m'
    WARNING = '\033[1;33m{}\033[0m'
    ERROR = '\033[1;31m{}\033[0m'
    

class Buffer:
    def __init__(self):
        pass

    def push(self, **kwargs):
        raise NotImplementedError
    


class EpisodesBufferEntry:
    """Entry for episode buffer"""
    def __init__(self):
        self.views = []
        self.features = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.terminal = False

    def append(self, view, feature, action, reward, alive, probs=None):
        self.views.append(view.copy())
        self.features.append(feature.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        if probs is not None:
            self.probs.append(probs)
        if not alive:
            self.terminal = True
            
            
class EpisodesBuffer(Buffer):
    """Replay buffer to store a whole episode for all agents
       one entry for one agent
    """
    def __init__(self, use_mean=False):
        super().__init__()
        self.buffer = {}
        self.use_mean = use_mean

    def push(self, **kwargs):
        view, feature = kwargs['state']
        acts = kwargs['acts']
        rewards = kwargs['rewards']
        alives = kwargs['alives']
        ids = kwargs['ids']

        if self.use_mean:
            probs = kwargs['prob']

        buffer = self.buffer
        index = np.random.permutation(len(view))

        for i in range(len(ids)):
            i = index[i]
            entry = buffer.get(ids[i])
            if entry is None:
                entry = EpisodesBufferEntry()
                buffer[ids[i]] = entry

            if self.use_mean:
                entry.append(view[i], feature[i], acts[i], rewards[i], alives[i], probs=probs[i])
            else:
                entry.append(view[i], feature[i], acts[i], rewards[i], alives[i])

    def reset(self):
        """ clear replay buffer """
        self.buffer = {}

    def episodes(self):
        """ get episodes """
        return self.buffer.values()

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
        
class SimpleAgentMemory(object):
    def __init__(self, obs_shape, feat_shape, act_n, max_len, use_mean=False):
        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.use_mean = use_mean

        if self.use_mean:
            self.prob = MetaBuffer((act_n,), max_len)

    def append(self, obs0, feat0, act, reward, alive, prob=None):
        self.obs0.append(np.array([obs0]))
        self.feat0.append(np.array([feat0]))
        self.actions.append(np.array([act], dtype=np.int32))
        self.rewards.append(np.array([reward]))
        self.terminals.append(np.array([not alive], dtype=bool))

        if self.use_mean:
            self.prob.append(np.array([prob]))

    def pull(self):
        res = {
            'obs0': self.obs0.pull(),
            'feat0': self.feat0.pull(),
            'act': self.actions.pull(),
            'rewards': self.rewards.pull(),
            'terminals': self.terminals.pull(),
            'prob': None if not self.use_mean else self.prob.pull()
        }

        return res


class SimpleMemoryGroup(object):
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
                self.agent[_id] = SimpleAgentMemory(self.obs_shape, self.feat_shape, self.act_n, self.sub_len, use_mean=self.use_mean)
            if self.use_mean:
                self.agent[_id].append(obs0=kwargs['state'][0][i], feat0=kwargs['state'][1][i], act=kwargs['acts'][i], reward=kwargs['rewards'][i], alive=kwargs['alives'][i], prob=kwargs['prob'][i])
            else:
                self.agent[_id].append(obs0=kwargs['state'][0][i], feat0=kwargs['state'][1][i], act=kwargs['acts'][i], reward=kwargs['rewards'][i], alive=kwargs['alives'][i])

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
            return obs, feature, actions, act_prob, obs_next, feature_next, act_next_prob, rewards, dones, masks
        else:
            return obs, feature, obs_next, feature_next, dones, rewards, actions, masks

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
        

class QmixReplayBuffer(object):
    def __init__(self, obs_shape, feat_shape, global_state_shape, act_n, max_len, batch_size, sub_len, use_mean=False):
        self.max_len = max_len
        self.batch_size = batch_size
        self.sub_len = sub_len
        self.use_mean = use_mean
        self.act_n = act_n
        num_agents = obs_shape[0]
        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((num_agents,), max_len, dtype='int32')
        self.rewards = MetaBuffer((num_agents,), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.masks = MetaBuffer((), max_len, dtype='bool')
        self.old_log_prob = MetaBuffer((num_agents,), max_len)
        if self.use_mean:
            self.prob = MetaBuffer((act_n,), max_len)
        self.global_states = MetaBuffer(global_state_shape, max_len)
        self._new_add = 0

    def push(self, **kwargs):
        self.obs0.append(np.array([kwargs['state'][0]]))    # (1, num_agents, H, W, C)
        self.feat0.append(np.array([kwargs['state'][1]]))     # (1, num_agents, feature_dim)
        self.actions.append(np.array([kwargs['acts']], dtype=np.int32))  # (1, num_agents)
        self.rewards.append(np.array([kwargs['rewards']]))     # (1, num_agents)
        terminal_flag = not all(kwargs['alives'])
        self.terminals.append(np.array([terminal_flag], dtype=bool))
        self.old_log_prob.append(np.array([kwargs['old_log_prob']]))
        if self.use_mean:
            self.prob.append(np.array([kwargs['prob']]))
        self.global_states.append(np.array([kwargs['global_state']]))
        mask = np.array([False if terminal_flag else True])
        self.masks.append(mask)
        self._new_add += 1

    def sample(self):
        idx = np.random.choice(self.nb_entries, size=self.batch_size)
        dones = self.terminals.data[idx]
        next_idx = np.where(dones, idx, (idx + 1) % self.nb_entries)
        
        obs = self.obs0.data[idx]
        obs_next = self.obs0.data[next_idx]
        feat = self.feat0.data[idx]
        feat_next = self.feat0.data[next_idx]
        actions = self.actions.data[idx]
        rewards = self.rewards.data[idx]
        masks = self.masks.data[idx]
        old_log_prob = self.old_log_prob.data[idx]
        if self.use_mean:
            act_prob = self.prob.data[idx]
            act_next_prob = self.prob.data[next_idx]
        else:
            act_prob = None
            act_next_prob = None
        global_state = self.global_states.data[idx]
        global_state_next = self.global_states.data[next_idx]
        return (obs, feat, obs_next, feat_next, dones, rewards, actions, masks, old_log_prob, global_state, global_state_next)
    def get_batch_num(self):
        buffer_length = len(self.obs0)
        print('\n[INFO] Qmix Replay Buffer length:', buffer_length, " new additions:", self._new_add)
        if buffer_length >= self.batch_size:
            res = buffer_length // self.batch_size
        else:
            res = 0
        self._new_add = 0
        return res
    @property
    def nb_entries(self):
        return len(self.obs0)