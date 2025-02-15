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
    
class QMixReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.size = args['buffer_size']
        self.episode_limit = args['max_episode_steps']
        self.num_agents = args['num_agents']
        self.num_actions = args['num_actions']
        self.obs_space = args['obs_space']
        self.state_space = args['state_space']
        
        self.buffers = {
            'o': np.empty([self.size, self.episode_limit, self.num_agents, self.obs_space]),
            'u': np.empty([self.size, self.episode_limit, self.num_agents, 1]),
            's': np.empty([self.size, self.episode_limit, self.state_space]),
            'r': np.empty([self.size, self.episode_limit, self.num_agents]),
            'o_next': np.empty([self.size, self.episode_limit, self.num_agents, self.obs_space]),
            's_next': np.empty([self.size, self.episode_limit, self.state_space]),
            'avail_u': np.empty([self.size, self.episode_limit, self.num_agents, self.num_actions]),
            'avail_u_next': np.empty([self.size, self.episode_limit, self.num_agents, self.num_actions]),
            'u_onehot': np.empty([self.size, self.episode_limit, self.num_agents, self.num_actions]),
            'padded': np.empty([self.size, self.episode_limit, 1]),
            'terminated': np.empty([self.size, self.episode_limit, 1])
        }
        
        self.current_idx = 0
        self.current_size = 0
        self.lock = threading.Lock()

    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx


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


class AgentMemory(object):
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
        print('\n[INFO] Length of buffer and new add:', len(self.obs0), self._new_add)
        res = self._new_add * 2 // self.batch_size
        self._new_add = 0
        return res

    @property
    def nb_entries(self):
        return len(self.obs0)
    

class MultiAgentMemoryGroup(object):
    def __init__(self, obs_shape, feat_shape, act_n, max_len, batch_size, sub_len, num_agents, use_mean=False):
        self.num_agents = num_agents
        self.agent_memories = [
            MemoryGroup(obs_shape, feat_shape, act_n, max_len, batch_size, sub_len, use_mean) 
            for _ in range(num_agents)
        ]
        self.max_len = max_len
        self.batch_size = batch_size
        self.use_mean = use_mean

    def push(self, **kwargs):
        # Assuming kwargs contain data for multiple agents
        for agent_id in range(self.num_agents):
            agent_kwargs = {
                'ids': [agent_id],
                'state': (kwargs['state'][0][agent_id:agent_id+1], kwargs['state'][1][agent_id:agent_id+1]),
                'acts': kwargs['acts'][agent_id:agent_id+1],
                'rewards': kwargs['rewards'][agent_id:agent_id+1],
                'alives': kwargs['alives'][agent_id:agent_id+1]
            }
            if self.use_mean:
                agent_kwargs['prob'] = kwargs['prob'][agent_id:agent_id+1]
            
            self.agent_memories[agent_id].push(**agent_kwargs)

    def tight(self):
        for agent_memory in self.agent_memories:
            agent_memory.tight()

    def sample(self):
        # Sample from each agent's memory
        samples = [memory.sample() for memory in self.agent_memories]
        
        # Transpose and stack the samples
        if self.use_mean:
            # For memories with probabilities
            obs = np.stack([s[0] for s in samples])
            feat = np.stack([s[1] for s in samples])
            acts = np.stack([s[2] for s in samples])
            probs = np.stack([s[3] for s in samples])
            obs_next = np.stack([s[4] for s in samples])
            feat_next = np.stack([s[5] for s in samples])
            probs_next = np.stack([s[6] for s in samples])
            rewards = np.stack([s[7] for s in samples])
            dones = np.stack([s[8] for s in samples])
            masks = np.stack([s[9] for s in samples])
            
            return obs, feat, acts, probs, obs_next, feat_next, probs_next, rewards, dones, masks
        else:
            # For memories without probabilities
            obs = np.stack([s[0] for s in samples])
            feat = np.stack([s[1] for s in samples])
            obs_next = np.stack([s[2] for s in samples])
            feat_next = np.stack([s[3] for s in samples])
            dones = np.stack([s[4] for s in samples])
            rewards = np.stack([s[5] for s in samples])
            acts = np.stack([s[6] for s in samples])
            masks = np.stack([s[7] for s in samples])
            
            return obs, feat, obs_next, feat_next, dones, rewards, acts, masks

    def get_batch_num(self):
        # Returns the minimum batch number across all agent memories
        return min(memory.get_batch_num() for memory in self.agent_memories)

import os
from moviepy.editor import ImageSequenceClip
class Runner(object):
    def __init__(self, env, handles, max_steps, models,
                play_handle, render_every=None, save_every=None, tau=0.001, 
                log_name=None, log_dir=None, model_dir=None, render_dir=None, 
                train=False, cuda=True):
        self.env = env
        self.models = models
        self.max_steps = max_steps
        self.handles = handles
        self.render_every = render_every
        self.save_every = save_every
        self.play = play_handle
        self.model_dir = model_dir
        self.render_dir = render_dir
        self.train = train
        self.tau = tau
        self.cuda = cuda
        self.log_dir = log_dir
        
        # Create all necessary directories
        self._create_directories()
        
    def _create_directories(self):
        """Create all necessary directories for logs, models, and renders."""
        directories = [self.log_dir, self.model_dir, self.render_dir]
        for directory in directories:
            if directory is not None:
                os.makedirs(directory, exist_ok=True)
                print(f"[INFO] Created directory: {directory}")

    def save_to_csv(self, round_losses, round_rewards, iteration):
        """Save round losses and rewards to a CSV file."""
        if self.log_dir is None:
            print("[WARNING] Log directory not specified, skipping CSV save")
            return
        data = {
            'Round': iteration,
            'Loss': round_losses[0][-1] if round_losses[0] else None,
            'Main Reward': round_rewards[0][-1] if round_rewards[0] else None,
            'Opponent Reward': round_rewards[1][-1] if round_rewards[1] else None,
        }
        
        df = pd.DataFrame([data])
        csv_path = os.path.join(self.log_dir, 'training_metrics.csv')
        
        try:
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, mode='w', header=True, index=False)
            print(f"[INFO] Successfully saved metrics to {csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save metrics to CSV: {str(e)}")
    
    def is_self_play(self):
        """Check if we're in self-play mode"""
        return hasattr(self.models[1], 'save')
    
    def soft_update(self):
        """Perform soft update from main model to opponent model in self-play"""
        if not self.is_self_play():
            return
            
        l_vars, r_vars = self.models[0].get_all_params(), self.models[1].get_all_params()
        for l_var, r_var in zip(l_vars, r_vars):
            r_var.detach().copy_((1. - self.tau) * l_var + self.tau * r_var)
            
    def run(self, variant_eps, iteration, win_cnt=None):
        info = {'main': None, 'opponent': None}
        info['main'] = {'ave_agent_reward': 0., 'total_reward': 0., 'kill': 0.}
        info['opponent'] = {'ave_agent_reward': 0., 'total_reward': 0., 'kill': 0.}
        
        max_nums, nums, mean_rewards, total_rewards, render_list = self.play(
            env=self.env, 
            n_round=iteration, 
            handles=self.handles,
            models=self.models, 
            print_every=50, 
            eps=variant_eps, 
            render=(iteration + 1) % self.render_every == 0 if self.render_every > 0 else False, 
            train=self.train, 
            cuda=self.cuda
        )
        
        for i, tag in enumerate(['main', 'opponent']):
            info[tag]['total_reward'] = total_rewards[i]
            info[tag]['kill'] = max_nums[i] - nums[1 - i]
            info[tag]['ave_agent_reward'] = mean_rewards[i]
            
        if self.train:
            print('\n[INFO] Main: {} \nOpponent: {}'.format(info['main'], info['opponent']))
            
            if info['main']['total_reward'] > info['opponent']['total_reward']:
                # Only perform soft update in self-play mode
                if self.is_self_play():
                    print('[INFO] Self-play mode - performing soft update...')
                    self.soft_update()
                    print('[INFO] Soft update completed')
                
                print('[INFO] Saving main model...')
                self.models[0].save(self.model_dir + '-main', iteration)
                
                # Save opponent model only in self-play mode
                if self.is_self_play():
                    print('[INFO] Saving opponent model...')
                    self.models[1].save(self.model_dir + '-opponent', iteration)
        else:
            if win_cnt is not None:
                if info['main']['kill'] > info['opponent']['kill']:
                    win_cnt['main'] += 1
                elif info['main']['kill'] < info['opponent']['kill']:
                    win_cnt['opponent'] += 1
                else:
                    win_cnt['main'] += 1
                    win_cnt['opponent'] += 1
        
        if len(render_list) > 0:
            print('[*] Saving Render')
            clip = ImageSequenceClip(render_list, fps=35)
            clip.write_gif('{}/replay_{}.gif'.format(self.render_dir, iteration+1), fps=20, verbose=False)
            print('[*] Saved Render')

            