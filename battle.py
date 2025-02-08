import os
import torch
import numpy as np
from magent2.environments import battle_v4 
from algo import spawn_ai
from senarios.senario_battle import play
import argparse

class RandomAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs, feature=None, prob=None, eps=0):
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        
        if len(obs.shape) > 3:
            return np.random.randint(0, self.num_actions, size=obs.shape[0])
        
        return np.array([np.random.randint(0, self.num_actions)])

    def train(self):
        pass

    def eval(self):
        pass

    def load(self, *args, **kwargs):
        pass

def run_battle_with_red_opponent(red_algo, blue_algo, red_step, blue_step, red_model_path, blue_model_path, render_dir, map_size=45, max_steps=400, use_cuda=True):
    # init env
    env = battle_v4.env(map_size=map_size, max_cycles=max_steps, render_mode="rgb_array")
    handles = env.unwrapped.env.get_handles()

    # load red model (handle random case)
    if red_algo == 'random':
        num_actions = env.unwrapped.env.get_action_space(handles[0])[0]
        red_model = RandomAgent(num_actions)
    else:
        red_model = spawn_ai(red_algo, env, handles[0], 'red', max_steps, use_cuda)
        if red_model_path:
            red_model.load(red_model_path, step=red_step)

    # load blue model (handle random case)
    if blue_algo == 'random':
        num_actions = env.unwrapped.env.get_action_space(handles[1])[0]
        blue_model = RandomAgent(num_actions)
    else:
        blue_model = spawn_ai(blue_algo, env, handles[1], 'blue', max_steps, use_cuda)
        if blue_model_path:
            blue_model.load(blue_model_path, step=blue_step)

    # run the battle
    render_dir = os.path.abspath(render_dir)
    os.makedirs(render_dir, exist_ok=True)
    render_path = os.path.join(render_dir, f"{red_algo}_vs_{blue_algo}.gif")

    _, _, _, _, render_list = play(
        env=env,
        n_round=0,
        handles=handles,
        models=[red_model, blue_model],
        print_every=50,
        eps=1.0,
        render=True,
        train=False,
        cuda=use_cuda
    )

    # save gif
    if render_list:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        print(f"[*] Saving render to {render_path}...")
        clip = ImageSequenceClip(render_list, fps=35)
        clip.write_gif(render_path, fps=35, verbose=False)
        print("[*] Render saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--red_algo', type=str, choices={'ac', 'mfac', 'mfq', 'iql', 'random', 'ppo'}, 
                       help='algorithm for main agent', required=True)
    parser.add_argument('--blue_algo', type=str, choices={'ac', 'mfac', 'mfq', 'iql', 'random', 'ppo'}, 
                       help='algorithm for opponent agent', required=True)
    parser.add_argument('--red_path', type=str,
                       help='path to red team model directory (required if not random)')
    parser.add_argument('--blue_path', type=str,
                       help='path to blue team model directory (required if not random)')
    parser.add_argument('--red_step', type=int, default=50, help='step of the pre-trained model in red') 
    parser.add_argument('--blue_step', type=int, default=50, help='step of the pre-trained model in blue')
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
    args = parser.parse_args()

    RENDER_DIR = "assets"  
    
    run_battle_with_red_opponent(
        red_algo=args.red_algo,
        blue_algo=args.blue_algo,
        red_step=args.red_step,
        blue_step=args.blue_step,
        red_model_path=args.red_path,
        blue_model_path=args.blue_path,
        render_dir=RENDER_DIR,
        map_size=80,
        max_steps=800,
        use_cuda=args.cuda 
    )