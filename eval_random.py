import os
import torch
import numpy as np
from magent2.environments import battle_v4 
from algo import spawn_ai
from senarios.senario_battle import play
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs, feature=None, prob=None, eps=0):
        # Nếu obs là tensor, chuyển sang numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        
        # Nếu obs có batch dimension
        if len(obs.shape) > 3:
            return np.array([self.action_space.sample() for _ in range(obs.shape[0])])
        
        # Trường hợp mặc định
        return np.array([self.action_space.sample()])

def run_battle_evaluation(algo, step, ac_model_path, render_dir, map_size=45, max_steps=300, use_cuda=True, num_episodes=10):
    blue_wins = 0
    red_wins = 0
    draws = 0
    blue_total_rewards = []
    red_total_rewards = []

    for episode in range(num_episodes):
        # env init
        env = battle_v4.env(map_size=map_size, max_cycles=max_steps)
        handles = env.unwrapped.env.get_handles()

        # load blue model
        blue_model = spawn_ai(algo, env, handles[0], 'blue', max_steps, use_cuda)
        blue_model.load(ac_model_path, step=step)

        # create random red model
        red_model = RandomAgent(env.action_space("red_0"))

        #RETURN: max_nums, nums, mean_rewards [mean_red, mean_blue], total_rewards, obs_list
        max_nums, nums, mean_rewards, _, obs_list = play(
            env=env,
            n_round=[episode],
            handles=handles,
            models=[red_model, blue_model],
            print_every=50,
            eps=1,
            render=False,
            train=False,
            cuda=use_cuda
        )

        n_dead = {
            "blue": max_nums[1] - nums[1],
            "red": max_nums[0] - nums[0]
        }

        # law
        who_loses = "red" if n_dead["red"] > n_dead["blue"] + 5 else "draw"
        who_loses = "blue" if n_dead["red"] + 5 < n_dead["blue"] else who_loses

        if who_loses == "red":
            blue_wins += 1
        elif who_loses == "blue":
            red_wins += 1
        else:
            draws += 1

        # mean rewards
        blue_total_rewards.append(mean_rewards[1])  
        red_total_rewards.append(mean_rewards[0])   

        # render for final episode
        if episode+1 == num_episodes:
            env = battle_v4.env(map_size=map_size, render_mode="rgb_array")
            handles = env.unwrapped.env.get_handles()
            
            blue_model = spawn_ai(algo, env, handles[0], 'blue', max_steps, use_cuda)
            blue_model.load(ac_model_path, step=step)
            
            red_model = RandomAgent(env.action_space("red_0"))

            _, _, _, _, render_list = play(
                env=env,
                n_round=[episode+1],
                handles=handles,
                models=[red_model, blue_model],
                print_every=50,
                eps=1.0,
                render=True,
                train=False,
                cuda=use_cuda
            )

            render_dir = os.path.abspath(render_dir)
            os.makedirs(render_dir, exist_ok=True)
            render_path = os.path.join(render_dir, "iql_vs_random.gif")
            
            if render_list:
                clip = ImageSequenceClip(render_list, fps=35)
                clip.write_gif(render_path, fps=35, verbose=False)
                print(f"[*] Render saved to {render_path}")

    # stat
    print("\n--- Evaluation Results ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"Blue Wins: {blue_wins} ({blue_wins/num_episodes*100:.2f}%)")
    print(f"Red Wins: {red_wins} ({red_wins/num_episodes*100:.2f}%)")
    print(f"Draws: {draws} ({draws/num_episodes*100:.2f}%)")
    print("\nReward Statistics:")
    print(f"Blue Average Reward: {np.mean(blue_total_rewards):.4f} ± {np.std(blue_total_rewards):.4f}")
    print(f"Red Average Reward: {np.mean(red_total_rewards):.4f} ± {np.std(red_total_rewards):.4f}")


if __name__ == "__main__":
    MODEL_PATH = "data/models/iql-0"
    RENDER_DIR = "data"
    ALGO = 'iql'
    STEP = 1999
    run_battle_evaluation(
        algo = ALGO,
        step = STEP,
        ac_model_path=MODEL_PATH,
        render_dir=RENDER_DIR,
        map_size=45,
        use_cuda=True,
        num_episodes=30
    )