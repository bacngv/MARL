import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ValueNet(nn.Module):
    def __init__(self, env, name, handle, update_every=5, use_mf=False, 
                 learning_rate=1e-4, polyak_tau=0.005, gamma=0.95, expectile=0.7):
        super(ValueNet, self).__init__()
        self.env = env
        self.name = name
        self._saver = None
        self.view_space = env.unwrapped.env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.unwrapped.env.get_feature_space(handle)[0]
        self.num_actions = env.unwrapped.env.get_action_space(handle)[0]
        self.update_every = update_every
        self.use_mf = use_mf  
        self.temperature = 0.1 
        self.lr = learning_rate
        self.polyak_tau = polyak_tau 
        self.gamma = gamma
        self.expectile = expectile  
        self.eval_net = self._construct_net()
        self.target_net = self._construct_net()
        self.optim = torch.optim.Adam(lr=self.lr, params=self.get_params(self.eval_net))
    def _construct_net(self):
        net = nn.ModuleDict()
        net['conv1'] = nn.Conv2d(in_channels=self.view_space[2], out_channels=32, kernel_size=3)
        net['conv2'] = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        flatten_dim = self.get_flatten_dim(net)
        net['obs_linear'] = nn.Linear(flatten_dim, 256)
        net['emb_linear'] = nn.Linear(self.feature_space, 32)
        if self.use_mf:
            net['prob_emb_linear'] = nn.Sequential(
                nn.Linear(self.num_actions, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
            shared_dim = 256 + 32 + 32  # 320
        else:
            shared_dim = 256 + 32  # 288
        net['q_head'] = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )
        net['v_head'] = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        return net
    def get_flatten_dim(self, net_dict):
        x = torch.zeros(1, self.view_space[2], self.view_space[0], self.view_space[1])
        x = F.relu(net_dict['conv1'](x))
        x = F.relu(net_dict['conv2'](x))
        return x.flatten().size(0)
    def get_params(self, net):
        params = []
        for k, v in net.items():
            params += list(v.parameters())
        return params
    def get_all_params(self):
        return self.get_params(self.eval_net) + self.get_params(self.target_net)
    def _shared(self, obs, feature, prob=None, net=None):
        if net is None:
            net = self.eval_net
        x = F.relu(net['conv1'](obs))
        x = F.relu(net['conv2'](x))
        x = x.flatten(start_dim=1)
        x = net['obs_linear'](x)
        y = net['emb_linear'](feature)
        if self.use_mf:
            z = net['prob_emb_linear'](prob)
            shared = torch.cat([x, y, z], dim=-1)
        else:
            shared = torch.cat([x, y], dim=-1)
        return shared
    def calc_target_q(self, obs_next, feature_next, dones, rewards):
        shared_next = self._shared(obs_next, feature_next, net=self.target_net)
        v_next = self.target_net['v_head'](shared_next).squeeze(-1)
        q_target = rewards + (1. - dones) * self.gamma * v_next
        return q_target
    def update(self):
        for k in self.target_net.keys():
            for param, target_param in zip(self.eval_net[k].parameters(), self.target_net[k].parameters()):
                target_param.detach().copy_(self.polyak_tau * param.detach() + (1. - self.polyak_tau) * target_param.detach())
    def act(self, obs, feature, prob=None, eps=None):
        beta = eps if eps is not None else self.temperature
        shared = self._shared(obs, feature, prob, net=self.eval_net)
        q = self.eval_net['q_head'](shared)
        v = self.eval_net['v_head'](shared)
        advantage = q - v 
        policy = F.softmax(advantage / beta, dim=-1)
        actions = policy.max(1)[1].detach().cpu().numpy()
        return actions
    def train_step(self, obs, feature, obs_next, feature_next, dones, rewards, acts, prob=None, mask=None):
        shared = self._shared(obs, feature, prob, net=self.eval_net)
        q = self.eval_net['q_head'](shared)
        v = self.eval_net['v_head'](shared).squeeze(-1)
        q_taken = torch.gather(q, 1, acts.unsqueeze(-1)).squeeze(-1)
        target_q = self.calc_target_q(obs_next, feature_next, dones, rewards)
        q_loss = F.mse_loss(q_taken, target_q.detach())
        # expectile regression
        error = q_taken - v
        weight = torch.where(error > 0, self.expectile, 1 - self.expectile)
        v_loss = (weight * (error ** 2)).mean()
        loss = q_loss + v_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item(), {'Q_loss': q_loss.item(), 'V_loss': v_loss.item(), 
                              'Q': q_taken.mean().item(), 'Target': target_q.mean().item()}
