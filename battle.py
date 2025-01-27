import os
import argparse
import torch
import numpy as np
from magent2.environments import battle_v4 
from algo import spawn_ai
from senarios.senario_battle import play

class RandomAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs, feature=None, prob=None, eps=0):
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        
        if len(obs.shape) > 3:
            return np.random.randint(0, self.num_actions, size=obs.shape[0])
        
        return np.array([np.random.randint(0, self.num_actions)])

    def load(self, *args, **kwargs):
        pass

    def eval(self):
        pass

def create_agent(algo, env, handle, name, max_steps, use_cuda, model_path=None, model_step=None):
    """Create either a random agent or load a trained model"""
    if algo == 'random':
        num_actions = env.env.get_action_space(handle)[0]
        return RandomAgent(num_actions)
    else:
        model = spawn_ai(algo, env, handle, name, max_steps, use_cuda)
        if model_path and model_step is not None:
            print(f"[*] Loading {name} model ({algo}) from: {model_path}")
            model.load(model_path, step=model_step)
        return model

def run_battle(red_algo, blue_algo, red_step, blue_step, red_model_path, blue_model_path, render_dir, map_size=45, max_steps=400, use_cuda=True):
    """
    Run battle between two agents (trained models or random)
    
    Args:
        red_algo: Algorithm type for red team (including 'random')
        blue_algo: Algorithm type for blue team (including 'random')
        red_step: Training step to load for red model (ignored if random)
        blue_step: Training step to load for blue model (ignored if random)
        red_model_path: Path to red team's model (ignored if random)
        blue_model_path: Path to blue team's model (ignored if random)
        render_dir: Directory to save the battle rendering
        map_size: Size of the battle map
        max_steps: Maximum steps per episode
        use_cuda: Whether to use CUDA
    """
    # Initialize environment
    env = battle_v4.env(map_size=map_size, max_cycles=max_steps, render_mode="rgb_array")
    handles = env.unwrapped.env.get_handles()

    # Create red agent (random or load model)
    red_model = create_agent(
        red_algo, env, handles[0], 'red', max_steps, use_cuda,
        red_model_path if red_algo != 'random' else None,
        red_step if red_algo != 'random' else None
    )

    # Create blue agent (random or load model)
    blue_model = create_agent(
        blue_algo, env, handles[1], 'blue', max_steps, use_cuda,
        blue_model_path if blue_algo != 'random' else None,
        blue_step if blue_algo != 'random' else None
    )

    # Set up render directory
    render_dir = os.path.abspath(render_dir)
    os.makedirs(render_dir, exist_ok=True)
    render_path = os.path.join(render_dir, f"{red_algo}_vs_{blue_algo}.gif")

    # Run the battle
    max_nums, nums, agent_r_records, total_rewards, render_list = play(
        env=env,
        n_round=0,
        handles=handles,
        models=[red_model, blue_model],
        print_every=50,
        eps=0.0,  # No exploration during evaluation
        render=True,
        train=False,
        cuda=use_cuda
    )

    # Calculate and print results
    red_kills = max_nums[0] - nums[1]
    blue_kills = max_nums[1] - nums[0]
    print(f"\nBattle Results:")
    print(f"Red Team ({red_algo}) Kills: {red_kills}")
    print(f"Blue Team ({blue_algo}) Kills: {blue_kills}")
    print(f"Red Team Reward: {total_rewards[0]:.2f}")
    print(f"Blue Team Reward: {total_rewards[1]:.2f}")

    # Save gif
    if render_list:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        print(f"[*] Saving render to {render_path}...")
        clip = ImageSequenceClip(render_list, fps=35)
        clip.write_gif(render_path, fps=35, verbose=False)
        print("[*] Render saved!")

    return red_kills, blue_kills, total_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--red_algo', type=str, choices={'ac', 'mfac', 'mfq', 'iql', 'random'}, 
                       help='algorithm for red team', required=True)
    parser.add_argument('--blue_algo', type=str, choices={'ac', 'mfac', 'mfq', 'iql', 'random'}, 
                       help='algorithm for blue team', required=True)
    parser.add_argument('--red_step', type=int, default=50, 
                       help='step of the pre-trained model for red team (ignored if random)')
    parser.add_argument('--blue_step', type=int, default=50, 
                       help='step of the pre-trained model for blue team (ignored if random)')
    parser.add_argument('--red_path', type=str,
                       help='path to red team model directory (required if not random)')
    parser.add_argument('--blue_path', type=str,
                       help='path to blue team model directory (required if not random)')
    parser.add_argument('--cuda', type=bool, default=True, 
                       help='use cuda')
    parser.add_argument('--map_size', type=int, default=80,
                       help='size of the battle map')
    parser.add_argument('--max_steps', type=int, default=800,
                       help='maximum steps per episode')
    args = parser.parse_args()

    # Validate arguments
    if args.red_algo != 'random' and not args.red_path:
        parser.error("--red_path is required when red_algo is not 'random'")
    if args.blue_algo != 'random' and not args.blue_path:
        parser.error("--blue_path is required when blue_algo is not 'random'")

    RENDER_DIR = "data/render"
    
    run_battle(
        red_algo=args.red_algo,
        blue_algo=args.blue_algo,
        red_step=args.red_step,
        blue_step=args.blue_step,
        red_model_path=args.red_path,
        blue_model_path=args.blue_path,
        render_dir=RENDER_DIR,
        map_size=args.map_size,
        max_steps=args.max_steps,
        use_cuda=args.cuda 
    )