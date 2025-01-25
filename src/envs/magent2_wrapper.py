import importlib
import gymnasium as gym
from gymnasium.spaces import Tuple
import magent2

class MAgent2Wrapper(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(self, env_name, **kwargs):
        # Import the specific MAgent2 environment dynamically
        self._env = magent2.make_env(env_name, **kwargs)
        
        obs, info = self._env.reset()
        self.n_agents = len(obs)
        self.last_obs = None

        # Create action and observation spaces as Tuple
        self.action_space = Tuple(
            tuple([self._env.action_space for _ in range(self.n_agents)])
        )
        self.observation_space = Tuple(
            tuple([self._env.observation_space for _ in range(self.n_agents)])
        )

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs = tuple(obs)
        self.last_obs = obs
        return obs, info

    def render(self, mode="human"):
        return self._env.render(mode)

    def step(self, actions):
        # Convert tuple actions to a format suitable for MAgent2
        observations, rewards, dones, truncated, infos = self._env.step(actions)

        # Convert observations and rewards to tuples
        obs = tuple(observations)
        rewards = list(rewards)

        # Check for episode termination
        done = all(dones)
        truncated = all(truncated)

        # If episode is done, use last observation
        if done:
            obs = self.last_obs
            rewards = [0] * len(obs)
        else:
            self.last_obs = obs

        # Flatten info dictionary if needed
        info = {
            f"agent_{i}_{key}": value
            for i, agent_info in enumerate(infos)
            for key, value in agent_info.items()
        }

        return obs, rewards, done, truncated, info

    def close(self):
        return self._env.close()

# Utility function to register MAgent2 environments with Gymnasium
def register_magent2_envs():
    # List of available MAgent2 environments
    magent2_envs = [
        "pursuit",
        "gather",
        "battle",
        # Add more environment names as needed
    ]

    for env_name in magent2_envs:
        gym.register(
            id=f"magent2-{env_name}-v4",
            entry_point="magent2_wrapper:MAgent2Wrapper",
            kwargs={
                "env_name": env_name,
            },
        )

# Call this function to register environments when the module is imported
register_magent2_envs()