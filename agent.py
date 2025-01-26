import time

import numpy as np
import torch
from qmix import QMIX
from env import get_epsilon


class Agents:
    def __init__(self, args, handel, models_names, logger):
        self.num_actions = args['num_actions']
        self.num_agents = args['num_agents']
        self.state_space = args['state_space']
        self.obs_space = args['obs_space']
        self.policy = QMIX(args, models_names, logger)
        self.args = args
        self.handel = handel
        self.models_names = models_names
        self.model_name = "qmix"
        self.logger = logger

    def choose_action(self, obs, last_action, epoch, evaluate,tag=False):
        if evaluate:
            epsilon = 0
        else:
            epsilon = get_epsilon(epoch, self.args['n_epoch'])
        if tag:
            epsilon=0
        inputs = obs.copy()
        if self.args['last_action']:
            inputs = np.hstack((inputs, last_action))


        hidden_state = self.policy.eval_hidden
        inputs = torch.tensor(inputs, dtype=torch.float32)


        if self.args['cuda']:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            inputs = inputs.cuda()
            hidden_state = hidden_state.to(device)

        q_value, self.policy.eval_hidden = self.policy.eval_rnn(inputs, hidden_state)


        if np.random.uniform() < epsilon:

            action = np.random.choice(self.args['num_actions'], size=(50, 1))
            action = action.astype(np.int32)
        else:
            if self.args['cuda']:
                max_values, action = torch.max(q_value, dim=1)
                action = action.cpu()
                action = action.numpy().astype(np.int32)
            else:
                max_values, action = torch.max(q_value, dim=1)
                action = action.numpy().astype(np.int32)

        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args['max_episode_steps']):
                if transition_idx + 1 >= max_episode_len:
                    max_episode_len = transition_idx + 1
                break
        return max_episode_len

    def train(self, batch, train_step):
        max_episdoe_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episdoe_len]
        train_time = self.policy.learn(batch, max_episdoe_len, train_step)
        return train_time
    def load_model(self):

        path_rnn = self.args['save_model_dir'] + '/' + '180g1_rnn_net_params.pkl'
        path_qmix = self.args['save_model_dir'] + '/' + '180g1_qmix_net_params.pkl'
        self.policy.eval_rnn.load_state_dict(torch.load(path_rnn))
        self.policy.eval_qmix_net.load_state_dict(torch.load(path_qmix))
        self.logger.info('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
        print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
