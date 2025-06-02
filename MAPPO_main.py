import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo import MAPPO_MPE
import os
from IPython import display as ipy_display
from matplotlib.ticker import FuncFormatter
from magent2.environments import battle_v4
from battleEnv import MAgent2Wrapper
from PIL import Image, ImageDraw, ImageFont
import io

class Runner_MAPPO_MAgent2:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set seaborn style
        sns.set_theme(style="whitegrid", font_scale=1.2)

        # Create MAgent2 Battle environment với wrapper
        self.env = MAgent2Wrapper(
            map_size=args.map_size,
            max_cycles=args.episode_limit,
            minimap_mode=args.minimap_mode,
            extra_features=args.extra_features,
            render_mode=None
        )
        
        # Reset environment to get initial observations
        observations, infos = self.env.reset(seed=self.seed)
        
        # Chọn số lượng agents để train (có thể chỉ train một phần)
        all_agents = list(observations.keys())
        
        # Chọn agents để train - có thể chỉ train red team hoặc cả hai team
        if args.train_team == "red":
            self.agent_list = self.env.red_agents[:args.num_agents]
        elif args.train_team == "blue":
            self.agent_list = self.env.blue_agents[:args.num_agents]
        elif args.train_team == "both":
            # Train cả hai team với số lượng agents bằng nhau
            num_per_team = args.num_agents // 2
            self.agent_list = (self.env.red_agents[:num_per_team] + 
                             self.env.blue_agents[:num_per_team])
        else:
            # Mặc định train tất cả agents
            self.agent_list = all_agents[:args.num_agents]
        
        self.args.N = len(self.agent_list)  # Number of training agents
        
        print(f"Training {self.args.N} agents: {self.agent_list}")
        
        # Get observation and action dimensions
        self.args.obs_dim_n = []
        self.args.action_dim_n = []
        
        for agent in self.agent_list:
            obs_space = self.env.observation_space(agent)
            action_space = self.env.action_space(agent)
            
            # MAgent2 observations are typically (height, width, channels)
            if len(obs_space.shape) == 3:
                obs_dim = np.prod(obs_space.shape)  # Flatten 3D observation
            else:
                obs_dim = obs_space.shape[0]
            
            self.args.obs_dim_n.append(obs_dim)
            self.args.action_dim_n.append(action_space.n)
        
        self.args.obs_dim = self.args.obs_dim_n[0]
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.state_dim = np.sum(self.args.obs_dim_n)
        
        print("Number of training agents:", self.args.N)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(
            self.env_name, self.number, self.seed))

        # Lists for evaluation rewards and steps
        self.evaluate_rewards = []
        self.eval_steps = []
        self.total_steps = 0
        
        # GIF saving variables
        self.gif_save_freq = 20000  # Save GIF every 20k steps
        self.last_gif_save = 0  # Track when last GIF was saved

        # Create folder for saving data if it doesn't exist
        os.makedirs('./data_train', exist_ok=True)
        os.makedirs('./gifs', exist_ok=True)  # Create folder for GIFs

        # Set up live plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        (self.line,) = self.ax.plot([], [], color='orange', label='MAPPO')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Episode Reward')
        self.ax.set_title('MAgent2 Battle')
        self.ax.legend(loc='lower right')
        self.fig.show()

        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self):
        evaluate_num = -1  # Number of evaluations performed
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1
            
            # Check if it's time to save GIF
            if self.total_steps - self.last_gif_save >= self.gif_save_freq:
                self.save_gif_episode()
                self.last_gif_save = self.total_steps

            _, episode_steps = self.run_episode_magent2(evaluate=False)
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()  # Final evaluation
        # Save final GIF
        self.save_gif_episode()
        self.env.close()
        self.save_eval_csv()
        plt.ioff()
        plt.show()

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_magent2(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward /= self.args.evaluate_times

        # Record evaluation steps and rewards
        self.eval_steps.append(self.total_steps)
        self.evaluate_rewards.append(evaluate_reward)

        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)

        # Save the model if necessary
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

        self.save_eval_csv()
        self.plot_eval_rewards()

    def save_eval_csv(self):
        csv_filename = './data_train/MAPPO_env_{}_number_{}_seed_{}.csv'.format(
            self.env_name, self.number, self.seed)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Training Steps', 'Evaluation Reward'])
            for step, reward in zip(self.eval_steps, self.evaluate_rewards):
                writer.writerow([step, reward])

    def plot_eval_rewards(self):
        # Update plot data
        self.line.set_xdata(self.eval_steps)
        self.line.set_ydata(self.evaluate_rewards)
        self.ax.relim()
        self.ax.autoscale_view()

        # Dynamic formatter for the X-axis
        def dynamic_formatter(x, pos):
            if x >= 1e6:
                return f'{x/1e6:.1f}M'
            elif x >= 1e3:
                return f'{x/1e3:.1f}K'
            else:
                return f'{int(x)}'
        self.ax.xaxis.set_major_formatter(FuncFormatter(dynamic_formatter))

        # Redraw canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Save the plot image
        plt.savefig('./data_train/MAPPO_env_{}_number_{}_seed_{}_eval.png'.format(
            self.env_name, self.number, self.seed))

    def save_gif_episode(self):
        """Save a GIF of one episode using the current policy"""
        print(f"Saving GIF at step {self.total_steps}...")
        
        # Create a separate environment for rendering
        render_env = MAgent2Wrapper(
            map_size=self.args.map_size,
            max_cycles=min(100, self.args.episode_limit),  # Shorter episodes for GIF
            minimap_mode=self.args.minimap_mode,
            extra_features=self.args.extra_features,
            render_mode='rgb_array'
        )
        
        frames = []
        observations, infos = render_env.reset(seed=self.seed)
        
        # Capture initial frame
        frame = render_env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))
        
        episode_step = 0
        
        while render_env.env.agents and episode_step < min(100, self.args.episode_limit):
            # Get observations for training agents only
            obs_n = []
            active_training_agents = []
            
            for agent in self.agent_list:
                if agent in observations:
                    obs = observations[agent]
                    # Flatten observation if it's 3D
                    if len(obs.shape) == 3:
                        obs = obs.flatten()
                    obs_n.append(obs)
                    active_training_agents.append(agent)
            
            if len(obs_n) == 0:
                break
            
            # Get actions for training agents using current policy
            if len(obs_n) == self.args.N:
                a_n, _ = self.agent_n.choose_action(obs_n, evaluate=True)
            else:
                # If some agents are dead, use random actions for simplicity
                a_n = [np.random.randint(0, self.args.action_dim) for _ in range(len(obs_n))]
            
            # Create action dictionary for all agents
            actions = {}
            
            # Add actions for training agents
            for i, agent in enumerate(active_training_agents):
                actions[agent] = a_n[i]
            
            # Add random actions for non-training agents
            for agent in observations:
                if agent not in actions:
                    action_space = render_env.action_space(agent)
                    actions[agent] = action_space.sample()
            
            # Step environment
            observations, rewards, terminations, truncations, infos = render_env.step(actions)
            
            # Capture frame
            frame = render_env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))
            
            episode_step += 1
            
            # Check if episode should end
            if (all(terminations.values()) or all(truncations.values()) or 
                episode_step >= min(100, self.args.episode_limit)):
                break
        
        render_env.close()
        
        # Save GIF
        if frames:
            gif_filename = f'./gifs/MAPPO_env_{self.env_name}_step_{self.total_steps}.gif'
            frames[0].save(
                gif_filename,
                save_all=True,
                append_images=frames[1:],
                duration=200,  # Duration per frame in milliseconds
                loop=0
            )
            print(f"GIF saved: {gif_filename}")
        else:
            print("No frames captured for GIF")

    def run_episode_magent2(self, evaluate=False):
        episode_reward = 0
        observations, infos = self.env.reset(seed=self.seed)
        
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
            
        episode_step = 0
        
        while self.env.env.agents and episode_step < self.args.episode_limit:
            # Get observations for training agents only
            obs_n = []
            active_training_agents = []
            
            for agent in self.agent_list:
                if agent in observations:
                    obs = observations[agent]
                    # Flatten observation if it's 3D (typical for MAgent2)
                    if len(obs.shape) == 3:
                        obs = obs.flatten()
                    obs_n.append(obs)
                    active_training_agents.append(agent)
            
            # If no training agents are active, break
            if len(obs_n) == 0:
                break
            
            # Pad observations if some training agents are dead
            while len(obs_n) < self.args.N:
                obs_n.append(np.zeros(self.args.obs_dim))
                
            # Get actions from trained agents
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n[:self.args.N], evaluate=evaluate)
            
            # Global state as concatenation of all observations
            s = np.concatenate(obs_n[:self.args.N])
            v_n = self.agent_n.get_value(s)
            
            # Create action dictionary for all agents
            actions = {}
            
            # Add actions for active training agents
            action_idx = 0
            for agent in active_training_agents:
                if action_idx < len(a_n):
                    actions[agent] = a_n[action_idx]
                    action_idx += 1
            
            # Add random actions for non-training agents (other team or inactive agents)
            for agent in observations:
                if agent not in actions:
                    action_space = self.env.action_space(agent)
                    actions[agent] = action_space.sample()
            
            # Step environment
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Get rewards for training agents only
            r_n = []
            done_n = []
            
            for agent in self.agent_list:
                if agent in rewards:
                    r_n.append(rewards[agent])
                    done_n.append(terminations.get(agent, False) or truncations.get(agent, False))
                else:
                    # Agent is dead/inactive
                    r_n.append(0.0)
                    done_n.append(True)
            
            # Pad rewards and done flags if needed
            while len(r_n) < self.args.N:
                r_n.append(0.0)
                done_n.append(True)
            
            # Calculate episode reward (sum of training agents' rewards)
            if r_n:
                episode_reward += np.sum(r_n[:len(active_training_agents)])

            # Store transition for training
            if not evaluate and len(obs_n) == self.args.N:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif self.args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            episode_step += 1
            
            # Check if episode should end
            if (all(terminations.values()) or all(truncations.values()) or 
                episode_step >= self.args.episode_limit):
                break

        # Store final value for advantage calculation
        if not evaluate and len(obs_n) == self.args.N:
            s = np.concatenate(obs_n[:self.args.N])
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step, v_n)

        return episode_reward, episode_step

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MAgent2 Battle environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=500, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Number of evaluations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Number of neurons in RNN hidden layers")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="Number of neurons in MLP hidden layers")
    parser.add_argument("--alliance_hidden_dim", type=int, default=64, help="Number of neurons in alliance hidden layers")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Use advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Use reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Use reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Policy entropy coefficient")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Use learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Use gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Use orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=bool, default=False, help="Use ReLU (if False, use tanh)")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=bool, default=False, help="Whether to add agent id")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="Whether to use value clipping")
    
    # MAgent2 specific arguments
    parser.add_argument("--map_size", type=int, default=15, help="Size of the battle map")
    parser.add_argument("--minimap_mode", type=bool, default=False, help="Use minimap observations")
    parser.add_argument("--extra_features", type=bool, default=False, help="Use extra features in observations")
    parser.add_argument("--num_agents", type=int, default=81, help="Number of agents to train")
    parser.add_argument("--train_team", type=str, default="red", choices=["red", "blue", "both"], 
                        help="Which team to train: red, blue, or both")

    args = parser.parse_args()
    runner = Runner_MAPPO_MAgent2(args, env_name="battle", number=1, seed=2)
    runner.run()