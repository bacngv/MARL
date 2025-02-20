import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import base
import replaybuffer


class DQN(base.ValueNet):
    def __init__(self, env, name, handle, sub_len, memory_size=2**10, batch_size=64, update_every=5, use_mf=False, learning_rate=0.0001, tau=0.005, gamma=0.95):
        super().__init__(env, name, handle, update_every=update_every, use_mf=use_mf, learning_rate=learning_rate, tau=tau, gamma=gamma)
        self.replay_buffer = replaybuffer.SimpleMemoryGroup(self.view_space, self.feature_space, self.num_actions, memory_size, batch_size, sub_len)
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)
    def train(self, cuda):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()
        total_loss = 0.0
        for i in range(batch_num):
            obs, feat, obs_next, feat_next, dones, rewards, acts, masks = self.replay_buffer.sample()
            obs = torch.FloatTensor(obs).permute([0, 3, 1, 2]).cuda() if cuda else torch.FloatTensor(obs).permute([0, 3, 1, 2])
            obs_next = torch.FloatTensor(obs_next).permute([0, 3, 1, 2]).cuda() if cuda else torch.FloatTensor(obs_next).permute([0, 3, 1, 2])
            feat = torch.FloatTensor(feat).cuda() if cuda else torch.FloatTensor(feat)
            feat_next = torch.FloatTensor(feat_next).cuda() if cuda else torch.FloatTensor(feat_next)
            acts = torch.LongTensor(acts).cuda() if cuda else torch.LongTensor(acts)
            rewards = torch.FloatTensor(rewards).cuda() if cuda else torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones).cuda() if cuda else torch.FloatTensor(dones)
            masks = torch.FloatTensor(masks).cuda() if cuda else torch.FloatTensor(masks)
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones)
            loss, q = super().train(obs=obs, feature=feat, target_q=target_q, acts=acts, mask=masks)
            self.update()
            total_loss += loss
            if i % 50 == 0:
                print('[*] IQL LOSS:', loss, '/ Q:', q)
        print('[*] IQL Total Loss:', total_loss)
        return total_loss
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        eval_file_path = os.path.join(dir_path, "dqn_eval_{}".format(step))
        target_file_path = os.path.join(dir_path, "dqn_target_{}".format(step))
        torch.save(self.eval_net.state_dict(), eval_file_path)
        torch.save(self.target_net.state_dict(), target_file_path)
        print("[*] Model saved")
    def load(self, dir_path, step=0):
        eval_file_path = os.path.join(dir_path, "dqn_eval_{}".format(step))
        target_file_path = os.path.join(dir_path, "dqn_target_{}".format(step))
        self.target_net.load_state_dict(torch.load(target_file_path))
        self.eval_net.load_state_dict(torch.load(eval_file_path))
        print("[*] Loaded model")
        
class MFQ(base.ValueNet):
    def __init__(self, env, name, handle, sub_len, eps=1.0, memory_size=2**10, batch_size=64, update_every=5, use_mf=True, learning_rate=0.0001, tau=0.005, gamma=0.95):
        super().__init__(env, name, handle, update_every=update_every, use_mf=use_mf, learning_rate=learning_rate, tau=tau, gamma=gamma)
        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            'act_n': self.num_actions,
            'use_mean': True,
            'sub_len': sub_len
        }
        self.train_ct = 0
        self.replay_buffer = replaybuffer.SimpleMemoryGroup(**config)
        self.update_every = update_every
        
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)
    def train(self, cuda):
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()
        total_loss = 0.0
        for i in range(batch_name):
            obs, feat, acts, act_prob, obs_next, feat_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()
            obs = torch.FloatTensor(obs).permute([0, 3, 1, 2]).cuda() if cuda else torch.FloatTensor(obs).permute([0, 3, 1, 2])
            obs_next = torch.FloatTensor(obs_next).permute([0, 3, 1, 2]).cuda() if cuda else torch.FloatTensor(obs_next).permute([0, 3, 1, 2])
            feat = torch.FloatTensor(feat).cuda() if cuda else torch.FloatTensor(feat)
            feat_next = torch.FloatTensor(feat_next).cuda() if cuda else torch.FloatTensor(feat_next)
            acts = torch.LongTensor(acts).cuda() if cuda else torch.LongTensor(acts)
            act_prob = torch.FloatTensor(act_prob).cuda() if cuda else torch.FloatTensor(act_prob)
            act_prob_next = torch.FloatTensor(act_prob_next).cuda() if cuda else torch.FloatTensor(act_prob_next)
            rewards = torch.FloatTensor(rewards).cuda() if cuda else torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones).cuda() if cuda else torch.FloatTensor(dones)
            masks = torch.FloatTensor(masks).cuda() if cuda else torch.FloatTensor(masks)
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones, prob=act_prob_next)
            loss, q = super().train(obs=obs, feature=feat, target_q=target_q, prob=act_prob, acts=acts, mask=masks)
            self.update()
            total_loss += loss
            if i % 50 == 0:
                print('[*] MFQ LOSS:', loss, '/ Q:', q)
        print('[*] MFQ Total Loss:', total_loss)
        return total_loss
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        eval_file_path = os.path.join(dir_path, "mfq_eval_{}".format(step))
        target_file_path = os.path.join(dir_path, "mfq_target_{}".format(step))
        torch.save(self.eval_net.state_dict(), eval_file_path)
        torch.save(self.target_net.state_dict(), target_file_path)
        print("[*] Model saved")
    def load(self, dir_path, step=0):
        eval_file_path = os.path.join(dir_path, "mfq_eval_{}".format(step))
        target_file_path = os.path.join(dir_path, "mfq_target_{}".format(step))
        self.target_net.load_state_dict(torch.load(target_file_path))
        self.eval_net.load_state_dict(torch.load(eval_file_path))
        print("[*] Loaded model")


class QMixNet(nn.Module):
    def __init__(self, num_agents, state_dim, mixing_embed_dim=32):
        super(QMixNet, self).__init__()
        self.num_agents = num_agents
        self.mixing_embed_dim = mixing_embed_dim
        self.hyper_w_1 = nn.Linear(state_dim, mixing_embed_dim * num_agents)
        self.hyper_w_final = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b_1 = nn.Linear(state_dim, mixing_embed_dim)
        self.V = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_qs, state):
        bs = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w_1(state))  # (bs, mixing_embed_dim * num_agents)
        w1 = w1.view(bs, self.num_agents, self.mixing_embed_dim)  # (bs, num_agents, mixing_embed_dim)
        b1 = self.hyper_b_1(state)  # (bs, mixing_embed_dim)
        b1 = b1.view(bs, 1, self.mixing_embed_dim)  # (bs, 1, mixing_embed_dim)
        agent_qs = agent_qs.view(bs, self.num_agents, 1)
        hidden = F.elu(torch.bmm(agent_qs.transpose(1, 2), w1) + b1)  # (bs, 1, mixing_embed_dim)
        w_final = torch.abs(self.hyper_w_final(state))  # (bs, mixing_embed_dim)
        w_final = w_final.view(bs, self.mixing_embed_dim, 1)  # (bs, mixing_embed_dim, 1)
        v = self.V(state).view(bs, 1, 1)  # (bs, 1, 1)
        y = torch.bmm(hidden, w_final) + v  # (bs, 1, 1)
        q_total = y.view(bs, 1)
        return q_total

class QMix(base.ValueNet):
    def __init__(self, env, name, handle, sub_len, num_agents, memory_size=2**10, batch_size=64,
                 update_every=5, learning_rate=0.0001, gamma=0.95, tau=0.01, use_mf=False, mixing_embed_dim=32):
        super().__init__(env, name, handle, update_every=update_every, use_mf=use_mf,
                         learning_rate=learning_rate, gamma=gamma)
        self.num_agents = num_agents
        local_obs_dim = np.prod(self.view_space)  # agent shape (H x W x C)
        local_feat_dim = self.feature_space
        global_state_dim = num_agents * (local_obs_dim + local_feat_dim)
        self.global_state_dim = global_state_dim
        self.gamma = gamma
        self.tau = tau
        self.use_mf = use_mf
        joint_obs_shape = (self.num_agents,) + self.view_space         # (num_agents, H, W, C)
        joint_feat_shape = (self.num_agents, self.feature_space)         # (num_agents, feature_dim)

        self.replay_buffer = replaybuffer.QmixReplayBuffer(joint_obs_shape, joint_feat_shape,
                                              (global_state_dim,), self.num_actions,
                                              memory_size, batch_size, sub_len, use_mean=use_mf)

        # agent network
        self.agent_net = self._construct_agent_net()
        self.target_agent_net = self._construct_agent_net()
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())

        # mixing network
        self.mixing_net = QMixNet(num_agents=self.num_agents, state_dim=self.global_state_dim,
                                  mixing_embed_dim=mixing_embed_dim)
        self.target_mixing_net = QMixNet(num_agents=self.num_agents, state_dim=self.global_state_dim,
                                         mixing_embed_dim=mixing_embed_dim)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        self.optimizer = torch.optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixing_net.parameters()),
            lr=learning_rate
        )

    def _construct_agent_net(self):
        net = nn.ModuleDict()
        net['conv1'] = nn.Conv2d(in_channels=self.view_space[2], out_channels=32, kernel_size=3)
        net['conv2'] = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        net['obs_linear'] = nn.Linear(self.get_flatten_dim(net), 256)
        net['emb_linear'] = nn.Linear(self.feature_space, 32)
        if self.use_mf:
            net['prob_emb_linear'] = nn.Sequential(
                nn.Linear(self.num_actions, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
        final_in_dim = 320 if self.use_mf else 288
        net['final_linear'] = nn.Sequential(
            nn.Linear(final_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )
        return net
    def get_agent_q_from_net(self, net, obs, feat, prob=None):
        x = F.relu(net['conv1'](obs))
        x = F.relu(net['conv2'](x))
        x = x.flatten(start_dim=1)
        x = net['obs_linear'](x)
        y = net['emb_linear'](feat)
        combined = torch.cat([x, y], dim=-1)
        if self.use_mf and (prob is not None):
            combined = torch.cat([combined, net['prob_emb_linear'](prob)], dim=-1)
        q_values = net['final_linear'](combined)  # (batch, num_actions)
        return q_values
    def get_agent_q(self, obs, feat, prob=None):
        return self.get_agent_q_from_net(self.agent_net, obs, feat, prob)
    def _get_target_agent_q(self, obs, feat, prob=None):
        return self.get_agent_q_from_net(self.target_agent_net, obs, feat, prob)
    def act(self, obs, feature, prob=None, eps=0.0):
        with torch.no_grad():
            feat_tensor = feature
            obs_tensor = obs
            q_values = self.get_agent_q(obs_tensor, feat_tensor)
            actions = q_values.argmax(dim=-1).cpu().numpy()
        return actions
    def update_target_network(self):
        for param, target_param in zip(self.agent_net.parameters(), self.target_agent_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.mixing_net.parameters(), self.target_mixing_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def flush_buffer(self, **kwargs):
        expected_agents = self.num_agents
        joint_obs, joint_feat = kwargs['state']
        current_agents = len(joint_obs)
        if current_agents < expected_agents:
            pad_size = expected_agents - current_agents
            # Padding 
            obs_shape = joint_obs[0].shape  # (H, W, C)
            pad_obs = [np.zeros(obs_shape, dtype=joint_obs.dtype) for _ in range(pad_size)]
            joint_obs = list(joint_obs) + pad_obs
            joint_obs = np.array(joint_obs)
            # Padding
            feat_shape = joint_feat[0].shape  # (feature_dim,)
            pad_feat = [np.zeros(feat_shape, dtype=joint_feat.dtype) for _ in range(pad_size)]
            joint_feat = list(joint_feat) + pad_feat
            joint_feat = np.array(joint_feat)
            kwargs['state'] = (joint_obs, joint_feat)
            # Padding
            if 'acts' in kwargs:
                acts = list(kwargs['acts'])
                acts += [0] * pad_size
                kwargs['acts'] = np.array(acts)
            # Padding 
            if 'rewards' in kwargs:
                rewards = list(kwargs['rewards'])
                rewards += [0] * pad_size
                kwargs['rewards'] = np.array(rewards)
        global_state = self.get_global_state(kwargs['state'][0], kwargs['state'][1])
        kwargs['global_state'] = global_state
        device = next(self.agent_net.parameters()).device
        obs_tensor = torch.FloatTensor(np.array(kwargs['state'][0])).permute(0, 3, 1, 2).to(device)
        feat_tensor = torch.FloatTensor(np.array(kwargs['state'][1])).to(device)
        if kwargs.get('prob') is not None:
            prob_tensor = torch.FloatTensor(np.array(kwargs['prob'])).to(device)
        else:
            prob_tensor = None
        acts_tensor = torch.LongTensor(kwargs['acts']).to(device)
        with torch.no_grad():
            _ = self.get_agent_q(obs_tensor, feat_tensor, prob_tensor)
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
        batch_num = self.replay_buffer.get_batch_num()
        if batch_num == 0:
            print("[QMIX] Not enough samples to train")
            return 0
        total_loss = 0
        for i in range(batch_num):
            sample_data = self.replay_buffer.sample()
            (obs, feat, obs_next, feat_next, dones, rewards,
            acts, masks, old_log_prob, global_state, global_state_next) = sample_data
            obs = torch.FloatTensor(obs)
            obs_next = torch.FloatTensor(obs_next)
            feat = torch.FloatTensor(feat)
            feat_next = torch.FloatTensor(feat_next)
            acts = torch.LongTensor(acts)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones.astype(np.float32))
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
                global_state = global_state.cuda()
                global_state_next = global_state_next.cuda()
            B, N, H, W, C = obs.shape
            obs_flat = obs.reshape(B * N, H, W, C).permute(0, 3, 1, 2)
            feat_flat = feat.reshape(B * N, feat.shape[-1])
            agent_q = self.get_agent_q(obs_flat, feat_flat)
            acts_flat = acts.reshape(-1)
            agent_q_chosen = agent_q.gather(1, acts_flat.unsqueeze(1)).view(B, N)
            q_total = self.mixing_net(agent_q_chosen, global_state)
            B_next, N_next, H_next, W_next, C_next = obs_next.shape
            obs_next_flat = obs_next.reshape(B_next * N_next, H_next, W_next, C_next).permute(0, 3, 1, 2)
            feat_next_flat = feat_next.reshape(B_next * N_next, feat_next.shape[-1])
            target_agent_q = self._get_target_agent_q(obs_next_flat, feat_next_flat)
            target_agent_q_max, _ = target_agent_q.max(dim=1)
            target_agent_q_max = target_agent_q_max.view(B, N)
            q_total_next = self.target_mixing_net(target_agent_q_max, global_state_next)
            reward_total = rewards.mean(dim=1)
            done_total = dones
            target_total = reward_total + self.gamma * q_total_next.squeeze(-1) * (1 - done_total)
            loss = F.mse_loss(q_total.squeeze(-1), target_total.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.update_target_network()
            if i % 50 == 0:
                print(f'[*] QMIX LOSS (Batch {i}): {loss.item():.4f} (MSE: {loss.item():.4f})')
        print(f'[*] TOTAL QMIX LOSS: {total_loss:.4f}')
        return total_loss
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        agent_path = os.path.join(dir_path, f"qmix_agent_net_{step}.pt")
        mix_path = os.path.join(dir_path, f"qmix_mixing_net_{step}.pt")
        torch.save(self.agent_net.state_dict(), agent_path)
        torch.save(self.mixing_net.state_dict(), mix_path)
        print("[*] QMIX model saved")
    def load(self, dir_path, step=0):
        agent_path = os.path.join(dir_path, f"qmix_agent_net_{step}.pt")
        mix_path = os.path.join(dir_path, f"qmix_mixing_net_{step}.pt")
        self.agent_net.load_state_dict(torch.load(agent_path))
        self.mixing_net.load_state_dict(torch.load(mix_path))
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        print("[*] QMIX model loaded")

