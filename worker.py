import copy
from env import *
import numpy as np
import time
import torch

leftID, rightID = 0, 1


class RolloutWorker:
    def __init__(self, args, logger):
        self.num_actions = args['num_actions']
        self.num_agents = args['num_agents']
        self.state_space = args['state_space']
        self.obs_space = args['obs_space']
        self.args = args
        self.epsilon = args['epsilon']
        self.min_epsilon = args['min_epsilon']
        self.logger = logger

    def generate_episode(self, Magentenv, agents_group1, agents_group2, epoch, evaluate):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
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

            if step >= self.args['max_episode_steps']:
                break

            group1_nums = Magentenv.env.get_num(agents_group1.handel)
            group2_nums = Magentenv.env.get_num(agents_group2.handel)

            group1_obs = Magentenv.get_obs(agents_group1.handel)  # 智能体数量*1048,死掉的用0填充

            group1_state = Magentenv.get_state(agents_group1.handel)
            actions, actions2, avail_actions, actions_onehot = [], [], [], []

            group1_action = agents_group1.choose_action(group1_obs, last_action_1, epoch + 1,evaluate)  # 当前智能体的动作值
            group1_action = group1_action.reshape(50, )
            group1_action_reward = group1_action[:group1_nums]
            action_onehot = np.zeros((self.num_agents, self.args['num_actions']))
            for id_index in range(self.num_agents):
                action_index = group1_action[id_index]
                action_onehot[id_index][action_index] = 1
            last_action_1 = action_onehot

            if test:
                # group2_obs = np.concatenate((group2_obs, np.zeros((self.num_agents - group2_nums, self.obs_space))),
                #                                 axis=0)  # 填充obs为50*obsspace
                group2_obs = Magentenv.get_obs(agents_group2.handel)  # 智能体数量*1048,死掉的用0填充
                group2_action = agents_group2.choose_action(group2_obs, last_action_2, epoch + 1,evaluate,tag=True)  # 当前智能体的动作值
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
            reward, terminated = Magentenv.step(agents_group1.handel)

            group1_nums = Magentenv.env.get_num(agents_group1.handel)
            group2_nums = Magentenv.env.get_num(agents_group2.handel)
            if step == 0:
                # 将结果拷贝到副本
                obs_copy = group1_obs
                action_copy = group1_action
                reward_copy = reward
                action_onehot_copy = action_onehot
            else:
                obs_copy[:group1_nums] = group1_obs[:group1_nums]
                action_copy[:group1_nums] = group1_action[:group1_nums]
                reward_copy[:group1_nums] = reward[:group1_nums]
                action_onehot_copy[:group1_nums] = action_onehot[:group1_nums]

            difference = self.num_agents - group1_nums
            # 填充不使用0填充经验，而是复制空缺的
            # 1、填充obs
            true_obs = group1_obs[:group1_nums]
            true_action = actions[:group1_nums]
            true_action_onehot = action_onehot[:group1_nums]
            if difference != 0:
                difference_action = action_copy[:difference]
                difference_obs = obs_copy[:difference]
                difference_action_onehot = action_onehot_copy[:difference]
                difference_reward = reward_copy[:difference]
                group1_obs = np.concatenate((true_obs, difference_obs), axis=0)
                actions = np.concatenate((true_action, difference_action), axis=0)
                action_onehot = np.concatenate((true_action_onehot, difference_action_onehot), axis=0)
                reward = np.concatenate((reward, difference_reward), axis=0)
            o.append(group1_obs)
            # 2、填充state
            s.append(group1_state)
            # 3、填充action
            u.append(np.reshape(actions, [self.num_agents, 1]))
            u_onehot.append(action_onehot)
            # avail_u.append(avail_actions)
            r.append(reward)
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            epsilon = get_epsilon(epoch, self.args['n_epoch'])
            if terminated == True or step == self.args['max_episode_steps']:
                end_time = time.time()
                group1_nums = Magentenv.env.get_num(agents_group1.handel)
                group2_nums = Magentenv.env.get_num(agents_group2.handel)

                if group1_nums > group2_nums and group2_nums <= 20:
                    # self.logger.info(' win!')
                    self.logger.info(
                        "train epoch {},group1剩余智能体数量:{}  group2剩余智能体数量:{}".format(epoch,group1_nums, group2_nums))
                    print(
                        "train epoch {},group1剩余智能体数量:{}  group2剩余智能体数量:{}".format(epoch,group1_nums, group2_nums))
                    if_win = True
                    #
                    # r[99] += 1
                else:
                    # self.logger.info(" lose!!")
                    self.logger.info(
                        "train epoch {},group1剩余智能体数量:{}  group2剩余智能体数量:{}".format(epoch,group1_nums, group2_nums))
                    print(
                        "train epoch {},group1剩余智能体数量:{}  group2剩余智能体数量:{}".format(epoch,group1_nums, group2_nums))
                    if_win = False
                # self.logger.info(" 对局用时：{}".format(end_time - start_time))
                break
            #######################
            Magentenv.env.clear_dead()
        o.append(group1_obs)
        s.append(group1_state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        for i in range(step, self.args['max_episode_steps']):  # 没有的字段用0填充，并且padded为1
            o.append(np.zeros((self.num_agents, self.obs_space)))
            u.append(np.zeros([self.num_agents, 1]))
            s.append(np.zeros(self.state_space))
            r.append(np.zeros(self.num_agents))
            o_next.append(np.zeros((self.num_agents, self.obs_space)))
            s_next.append(np.zeros(self.state_space))
            u_onehot.append(np.zeros((self.num_agents, self.num_actions)))
            padded.append([1.])
            terminate.append([1.])

        '''
        (o[n], u[n], r[n], o_next[n], avail_u[n], u_onehot[n])组成第n条经验，各项维度都为(episode数，transition数，n_agents, 自己的具体维度)
         因为avail_u表示当前经验的obs可执行的动作，但是计算target_q的时候，需要obs_net及其可执行动作，
        '''
        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       # avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        # if not evaluate:
        #     self.epsilon = epsilon

        return episode, episode_reward, if_win
