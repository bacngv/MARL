import os
import torch
import numpy as np
from magent2.environments import battle_v4 
from algo import spawn_ai
from senarios.senario_battle import play
from torch_model import QNetwork

def run_battle_with_red_opponent(algo, step, ac_model_path, red_model_path, render_dir, map_size=45, max_steps=400, use_cuda=True):
    # init env
    env = battle_v4.env(map_size=map_size, max_cycles = max_steps, render_mode="rgb_array")
    handles = env.unwrapped.env.get_handles()

    # load ac or mfac pretrained model 
    blue_model = spawn_ai(algo, env, handles[0], 'blue', max_steps, use_cuda)
    blue_model.load(ac_model_path, step=step)  

    # load red.pt
    q_network = QNetwork(
        env.observation_space("red_0").shape, 
        env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load(red_model_path, weights_only=True, map_location="cpu")
    )

    # create a wrapper for the q-learning 
    class QNetworkWrapper:
        def __init__(self, q_network, use_cuda=True):
            self.q_network = q_network.cuda() if use_cuda else q_network
            self.num_actions = q_network.network[-1].out_features
            self.use_cuda = use_cuda

        def act(self, obs, feature=None, prob=None, eps=0):
            # Ensure obs is on the correct device
            if self.use_cuda:
                obs = obs.cuda()
            
            # Add batch dimension if needed
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)
            
            # Use Q-network for action selection
            with torch.no_grad():
                q_values = self.q_network(obs)
                return torch.argmax(q_values, dim=1).cpu().numpy()
    red_model = QNetworkWrapper(q_network)

    # run the battle
    render_dir = os.path.abspath(render_dir)
    os.makedirs(render_dir, exist_ok=True)
    render_path = os.path.join(render_dir, "battle_vs_dqn.gif")

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
    RED_MODEL_PATH = "red.pt"  
    RENDER_DIR = "data"  
    MODEL_NAME = 'iql'
    STEP = 1999
    run_battle_with_red_opponent(
        algo=MODEL_NAME,
        step=STEP,
        ac_model_path=AC_MODEL_PATH,
        red_model_path=RED_MODEL_PATH,
        render_dir=RENDER_DIR,
        map_size=45,
        max_steps=300,
        use_cuda=True 
    )
