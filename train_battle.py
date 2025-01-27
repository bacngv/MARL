"""Battle Training with Flexible Opponent Selection"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from magent2.environments import battle_v4
from algo import spawn_ai
from algo import tools
from senarios.senario_battle import play

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs('./data', exist_ok=True)

class RandomAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs, feature=None, prob=None, eps=0):
        # Convert tensor to numpy if needed
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        
        # Handle batch dimension
        if len(obs.shape) > 3:
            return np.random.randint(0, self.num_actions, size=obs.shape[0])
        
        # Default case
        return np.array([np.random.randint(0, self.num_actions)])

    def save(self, dir_path, step):
        pass

    def train(self):
        pass

    def eval(self):
        pass

def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]
    if epoch == start:
        return min_v
    eps = min_v
    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break
    return eps

def create_opponent(opponent_type, env, handles, max_steps, cuda):
    """Create opponent based on specified type"""
    if opponent_type == 'random':
        num_actions = env.env.get_action_space(handles[1])[0]
        return RandomAgent(num_actions)
    else:
        return spawn_ai(opponent_type, env, handles[1], f'{opponent_type}-opponent', max_steps, cuda)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'iql'}, help='algorithm for main agent', required=True)
    parser.add_argument('--opponent', type=str, choices={'random', 'ac', 'mfac', 'mfq', 'iql'}, help='opponent type (random or algorithm)', required=True)
    parser.add_argument('--save_every', type=int, default=20, help='decide the save interval')
    parser.add_argument('--update_every', type=int, default=5, help='decide the update interval for q-learning, optional')
    parser.add_argument('--n_round', type=int, default=600, help='set the training round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=80, help='set the size of map') 
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
    args = parser.parse_args()

    # Initialize environment
    env = battle_v4.env(map_size=args.map_size, max_cycles=args.max_steps, render_mode="rgb_array")
    env = env.unwrapped  # Get the underlying environment
    handles = env.env.get_handles()

    # Create directory paths based on main algorithm and opponent type
    base_log_dir = os.path.join(BASE_DIR, 'data/tmp/{}_{}'.format(args.algo, args.opponent))
    base_render_dir = os.path.join(BASE_DIR, 'data/render/{}_{}'.format(args.algo, args.opponent))
    base_model_dir = os.path.join(BASE_DIR, 'data/models/{}_{}'.format(args.algo, args.opponent))
    
    start_from = 0
    
    # Create main agent
    main_model = spawn_ai(args.algo, env, handles[0], args.algo + '-main', args.max_steps, args.cuda)
    
    # Create opponent (random or algorithm-based)
    opponent_model = create_opponent(args.opponent, env, handles, args.max_steps, args.cuda)
    
    models = [main_model, opponent_model]

    runner = tools.Runner(
        env, 
        handles, 
        args.max_steps, 
        models, 
        play,
        render_every=args.save_every if args.render else 0, 
        save_every=args.save_every, 
        log_name=f"{args.algo}_{args.opponent}",
        log_dir=base_log_dir, 
        model_dir=base_model_dir, 
        render_dir=base_render_dir, 
        train=True, 
        cuda=args.cuda
    )

    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        runner.run(eps, k)
