import gymnasium as gym
from gymnasium.spaces import Tuple
import magent2.environments

class MAgent2Wrapper(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(self, env_name, **kwargs):
        # Dynamic import might require a different approach
        if env_name == "battle":
            self._env = magent2.environments.battle.parallel_env(**kwargs)
        elif env_name == "pursuit":
            self._env = magent2.environments.pursuit.parallel_env(**kwargs)
        elif env_name == "gather":
            self._env = magent2.environments.gather.parallel_env(**kwargs)
        else:
            raise ValueError(f"Unsupported environment: {env_name}")
        
        obs, info = self._env.reset()
        self.n_agents = len(obs)
        self.last_obs = None

        self.action_space = Tuple(
            tuple([self._env.action_space(agent) for agent in self._env.agents])
        )
        self.observation_space = Tuple(
            tuple([self._env.observation_space(agent) for agent in self._env.agents])
        )

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs = tuple(obs.values())
        self.last_obs = obs
        return obs, info

    def step(self, actions):
        observations, rewards, dones, truncated, infos = self._env.step({agent: action for agent, action in zip(self._env.agents, actions)})

        obs = tuple(observations.values())
        rewards = list(rewards.values())

        done = all(dones.values())
        truncated = all(truncated.values())

        if done:
            obs = self.last_obs
            rewards = [0] * len(obs)
        else:
            self.last_obs = obs

        info = {
            f"agent_{i}_{key}": value
            for i, agent_info in enumerate(infos.values())
            for key, value in agent_info.items()
        }

        return obs, rewards, done, truncated, info

    def close(self):
        return self._env.close()

def register_magent2_envs():
    magent2_envs = ["pursuit", "gather", "battle"]

    for env_name in magent2_envs:
        gym.register(
            id=f"magent2-{env_name}-v4",
            entry_point="envs.magent2_wrapper:MAgent2Wrapper",
            kwargs={"env_name": env_name},
        )

register_magent2_envs()