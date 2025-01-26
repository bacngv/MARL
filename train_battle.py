import torch
import numpy as np
from magent2.environments import battle_v4
from qmix import QMIX
from replay_buffer import ReplayBuffer
import yaml

def load_args_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return data['args']  # Access the 'args' dictionary directly


def preprocess_env(env):
    env.reset()
    env_agents = env.possible_agents
    obs_spaces = env.observation_space(env_agents[0]).shape[0]
    state_spaces = obs_spaces * len(env_agents)
    num_actions = env.action_space(env_agents[0]).n
    return obs_spaces, state_spaces, num_actions, env_agents

def run_episode(env, agents, qmix, buffer, args):
    env.reset()
    terminated = False
    episode_steps = 0
    obs_dict = {agent: env.observe(agent) for agent in agents}
    obs = np.array([obs_dict[agent] for agent in agents])

    episode_batch = {
        'o': [],
        'u': [],
        'r': [],
        'o_next': [],
        'terminated': [],
    }
    
    while not terminated:
        actions = []
        for agent_id in range(len(agents)):
            obs_agent = torch.tensor(obs[agent_id], dtype=torch.float32).unsqueeze(0)
            if args['cuda']:
                obs_agent = obs_agent.cuda()
            q_values, _ = qmix.eval_rnn(obs_agent, qmix.eval_hidden[agent_id].unsqueeze(0))
            action = torch.argmax(q_values).item()
            actions.append(action)

        action_dict = {agents[i]: actions[i] for i in range(len(agents))}
        next_obs_dict, rewards, dones, _ = env.step(action_dict)
        next_obs = np.array([next_obs_dict[agent] for agent in agents])
        reward = np.array([rewards[agent] for agent in agents])
        terminated = any(dones.values())

        episode_batch['o'].append(obs)
        episode_batch['u'].append(actions)
        episode_batch['r'].append(reward)
        episode_batch['o_next'].append(next_obs)
        episode_batch['terminated'].append([terminated] * len(agents))

        obs = next_obs
        episode_steps += 1
        if episode_steps >= args['max_episode_steps']:
            break
    
    buffer.store_episode(episode_batch)

def train_qmix(qmix, buffer, args):
    if buffer.current_size < args['batch_size']:
        return
    batch = buffer.sample(args['batch_size'])
    qmix.learn(batch, args['max_episode_steps'], args['train_step'])
    args['train_step'] += 1

def main():
    # Load configuration from YAML
    args = load_args_from_yaml("/content/MARL/application.yaml")

    # Make sure to update certain values that might not be in the YAML file
    env = battle_v4.env(map_size=45, render_mode=None)
    obs_spaces, state_spaces, num_actions, agents = preprocess_env(env)

    args.update({
        'obs_space': obs_spaces,
        'state_space': state_spaces,
        'num_actions': num_actions
    })

    buffer = ReplayBuffer(args)
    qmix = QMIX(args, ["qmix_net", "rnn_net"], logger=None)

    num_episodes = 5000
    for episode in range(num_episodes):
        run_episode(env, agents, qmix, buffer, args)
        train_qmix(qmix, buffer, args)

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed")

if __name__ == "__main__":
    main()
