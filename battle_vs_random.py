import os
import torch
import numpy as np
from magent2.environments import battle_v4 
from algo import spawn_ai
from senarios.senario_battle import play

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

def run_battle_with_random_opponent(algo, step, ac_model_path, render_dir, map_size=45, max_steps=400, use_cuda=True):
    # init env
    env = battle_v4.env(map_size=map_size, max_cycles = max_steps, render_mode="rgb_array")
    handles = env.unwrapped.env.get_handles()

    # load ac or mfac pretrained model 
    blue_model = spawn_ai(algo, env, handles[0], 'blue', max_steps, use_cuda)
    blue_model.load(ac_model_path, step=step)  

    # tạo agent random cho red
    red_model = RandomAgent(env.action_space("red_0"))

    # run the battle
    render_dir = os.path.abspath(render_dir)
    os.makedirs(render_dir, exist_ok=True)
    render_path = os.path.join(render_dir, "battle_vs_random.gif")

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
    AC_MODEL_PATH = "data/models/iql-0"  
    RENDER_DIR = "data"  
    MODEL_NAME = 'iql'
    STEP = 1999
    run_battle_with_random_opponent(
        algo=MODEL_NAME,
        step=STEP,
        ac_model_path=AC_MODEL_PATH,
        render_dir=RENDER_DIR,
        map_size=45,
        max_steps=300,
        use_cuda=True 
    )