import torch
import os
from qmix_net import RNN
from qmix_net import QMixNet
import time


class QMIX:
    def __init__(self, args, models_names, logger):
        self.num_actions = args['num_actions']
        self.num_agents = args['num_agents']
        self.state_space = args['state_space']
        self.obs_space = args['obs_space']
        input_shape = self.obs_space
        self.logger = logger
        self.eval_hidden = None
        if args['last_action']:
            input_shape += self.num_actions
        # if args.reuse_network:
        #     input_shape += self.num_agents
        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)
        self.target_qmix_net = QMixNet(args)
        self.args = args
        if self.args['cuda']:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.eval_rnn.to(device)
            self.target_rnn.to(device)
            self.eval_qmix_net.to(device)
            self.target_qmix_net.to(device)
        self.save_model_dir = args['save_model_dir']
        self.load_model_dir = args['load_model_dir']

        if self.args['load_model']:
            if os.path.exists(self.load_model_dir):
                path_rnn = self.load_model_dir + '/' + models_names[1]
                path_qmix = self.load_model_dir + '/' + models_names[0]
                self.eval_rnn.load_state_dict(torch.load(path_rnn))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix))
                self.logger.info('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args['optimizer'] == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=float(args['lr']))
        self.target_hidden = None

    def learn(self, batch, max_episode_len, train_step,):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        learn_time=time.time()
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
            batch['r'], batch['avail_u'], batch['avail_u_next'], \
            batch['terminated']
        mask = 1 - batch["padded"].float()

        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        if self.args['cuda']:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            s = s.to(device)
            u = u.to(device)
            r = r.to(device)
            s_next = s_next.to(device)
            terminated = terminated.to(device)
            mask = mask.to(device)
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        targets = r + self.args['gamma'] * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args['grad_norm_clip'])
        self.optimizer.step()
        backward_time = time.time() - learn_time
        if train_step > 0 and train_step % self.args['target_update_cycle'] == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        return backward_time

    def _get_inputs(self, batch, transition_idx):
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
            batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        if self.args['last_action']:
            if transition_idx == 0:  # 첫 경험이라면, 이전 행동을 0 벡터로하십시오
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        inputs = torch.cat([x.reshape(episode_num * self.args['num_agents'], -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args['num_agents'], -1) for x in inputs_next], dim=1)

        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            if self.args['cuda']:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                inputs = inputs.to(device)
                inputs_next = inputs_next.to(device)
                self.eval_hidden = self.eval_hidden.to(device)
                self.target_hidden = self.target_hidden.to(device)
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(8, ,num_actions)
            q_eval = q_eval.view(episode_num, self.num_agents, -1)
            q_target = q_target.view(episode_num, self.num_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.num_agents, self.args['rnn_hidden_dim']))
        self.target_hidden = torch.zeros((episode_num, self.num_agents, self.args['rnn_hidden_dim']))

    def save_model(self, train_step):
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.save_model_dir + '/' + str(train_step)+'g1_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.save_model_dir + '/' +str(train_step)+'g1_rnn_net_params.pkl')
