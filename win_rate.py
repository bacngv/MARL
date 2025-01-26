import copy
from env import *
import numpy as np
import time
import torch

leftID, rightID = 0, 1


class WinRate:
    def __init__(self, args, logger):
        # self.episode_limit = args['episode_limit
        self.num_actions = args['num_actions']
        self.num_agents = args['num_agents']
        self.state_space = args['state_space']
        self.obs_space = args['obs_space']
        self.args = args
        self.epsilon = args['epsilon']
        self.min_epsilon = args['min_epsilon']
        self.logger = logger

    def generate_episode(self, Magentenv, episode_num, agents_group1, agents_group2, epoch, evaluate):

        terminated = False
        if_win = False
        step = 0
        episode_reward = 0

        last_action_1 = np.zeros((self.num_agents, self.num_actions))
        last_action_2 = np.zeros((self.num_agents, self.num_actions))
        agents_group1.policy.init_hidden(1)
        test = agents_group2.model_name == "qmix"

        if test:
            agents_group2.policy.init_hidden(1)
        elif agents_group2.model_name == "dqn":
            pass

        while not terminated:

            group1_nums = Magentenv.env.get_num(agents_group1.handel)
            group2_nums = Magentenv.env.get_num(agents_group2.handel)

            group1_obs = Magentenv.get_obs(agents_group1.handel)  # 智能体数量*1048,死掉的用0填充
            group1_state = Magentenv.get_state(agents_group1.handel)
            actions, actions2, avail_actions, actions_onehot = [], [], [], []

            group1_action = agents_group1.choose_action(group1_obs, last_action_1, epoch + 1,
                                                        evaluate=True)  # 当前智能体的动作值
            group1_action = group1_action.reshape(50, )
            group1_action_reward = group1_action[:group1_nums]
            action_onehot = np.zeros((self.num_agents, self.args['num_actions']))
            for id_index in range(self.num_agents):
                action_index = group1_action[id_index]
                action_onehot[id_index][action_index] = 1
            last_action_1 = action_onehot

            if test:
                group2_obs = Magentenv.get_obs(agents_group2.handel)  # 智能体数量*1048,死掉的用0填充
                group2_action = agents_group2.choose_action(group2_obs, last_action_2, epoch + 1, evaluate=False,tag=True)  # 当前智能体的动作值
                action_onehot = np.zeros((self.num_agents, self.args['num_actions']))
                for id_index in range(self.num_agents):
                    action_index = group2_action[id_index]
                    action_onehot[id_index][action_index] = 1
                last_action_2 = action_onehot
            else:
                group2_obs = Magentenv.env.get_observation(agents_group2.handel)

                group2_action = agents_group2.choose_action(group2_obs, k=step, evaluate=False)
                actions2.append(group2_action)

            actions = group1_action
            Magentenv.env.set_action(agents_group1.handel, group1_action_reward)  #
            Magentenv.env.set_action(agents_group2.handel, group2_action)  #
            reward, terminated= Magentenv.step(agents_group1.handel)
            episode_reward += np.sum(reward)
            step += 1
            group1_nums = Magentenv.env.get_num(agents_group1.handel)
            group2_nums = Magentenv.env.get_num(agents_group2.handel)

            epsilon = get_epsilon(epoch, self.args['n_epoch'])

            if terminated == True or step >= self.args['max_episode_steps_test']:

                if group1_nums > group2_nums:
                    # self.logger.info(' win!')
                    self.logger.info(
                        "train epoch {},group1剩余智能体数量:{}  group2剩余智能体数量:{},win!".format(epoch,
                                                                                                      group1_nums,
                                                                                                      group2_nums))
                    print(
                        "train epoch {},group1剩余智能体数量:{}  group2剩余智能体数量:{},win!".format(epoch,
                                                                                                      group1_nums,
                                                                                                      group2_nums))
                    if_win = True
                    # for i in range(len(r)):
                    #     r[0] += 100
                else:
                    # self.logger.info(" lose!!")
                    self.logger.info(
                        "train epoch {},group1剩余智能体数量:{}  group2剩余智能体数量:{},lose!".format(epoch,
                                                                                                       group1_nums,
                                                                                                       group2_nums))
                    print(
                        "train epoch {},group1剩余智能体数量:{}  group2剩余智能体数量:{},lose!".format(epoch,
                                                                                                       group1_nums,
                                                                                                       group2_nums))
                    if_win = False
                # self.logger.info(" 对局用时：{}".format(end_time - start_time))
                break
            Magentenv.env.render()
            Magentenv.env.clear_dead()

        return if_win, episode_reward
