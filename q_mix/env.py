import argparse

import math
import numpy as np
import magent

# import torch

epsilon_end = 0.05
epsilon_start = 0.75
epsilon_decay = 2 # epsilon衰减率


def get_epsilon(current_step, total_steps):
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
              math.exp(-epsilon_decay * current_step / total_steps)
    return epsilon




class MagentEnv:
    def __init__(self, args):
        self.env = magent.GridWorld("battle", map_size=args['map_size'])
        self.handels = self.env.get_handles()
        self.args = args
        self.action_space = 21
        self.obs_space = args['obs_space']
        self.state_space = args['state_space']
        self.num_agent = args['num_agents']
        self.current_step = 0
        self.observation = None
        self.state = None

    def reset(self, handel):
        self.current_step = 0
        self.done = False
        # self.env.reset()
        return self.get_obs(handel), self.get_state(handel)



    def step(self,handel):
        done = self.env.step()
        self.current_step = self.current_step + 1
        reward = self.env.get_reward(handel)
        return reward, done

    # ====== RUN ======
    def get_obs(self, handle):
        q = []
        agent_list = []
        obs = self.env.get_observation(handle)
        feather = obs[1]
        view = obs[0]
        alive_agents_num = self.env.get_num(handle)

        for agent_id in range(self.args['num_agents']):
            if agent_id < alive_agents_num:
                agent_feather = feather[agent_id]
                agent_view = view[agent_id]
                for t in range(9):
                    for j in range(9):
                        for k in range(7):
                            q.append(agent_view[t][j][k])
                z = np.array(q)
                q = []
                agent_obs = np.concatenate((z, agent_feather))
                agent_list.append(agent_obs)
            else:
                zero = np.zeros(self.args['obs_space'])
                agent_list.append(zero)
        agent_numpy = np.array(agent_list)
        observation = agent_numpy.astype(float)

        return observation

    def set_action(self, handle, actions):
        self.env.set_action(handle, actions)

    def get_state(self, handle):  # shape:obs_space*agent_num

        group1_nums = self.env.get_num(handle)
        difference = self.num_agent - group1_nums
        obs = self.get_obs(handle)
        obs = obs[:15, :]
        state = obs.flatten()
        return state

