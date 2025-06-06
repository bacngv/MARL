import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from torch.utils.tensorboard import SummaryWriter
import argparse
from replay_buffer import ReplayBuffer
from qmix import QMIX
from normalization import Normalization
import os
from IPython import display as ipy_display
from matplotlib.ticker import FuncFormatter
from magent2.environments import battle_v4
from battleEnv import MAgent2Wrapper
from PIL import Image, ImageDraw, ImageFont
import io


class Runner_QMIX_MAgent2:
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
        obs_dims = []
        action_dims = []
        
        for agent in self.agent_list:
            obs_space = self.env.observation_space(agent)
            action_space = self.env.action_space(agent)
            
            # MAgent2 observations are typically (height, width, channels)
            if len(obs_space.shape) == 3:
                obs_dim = np.prod(obs_space.shape)  # Flatten 3D observation
            else:
                obs_dim = obs_space.shape[0]
            
            obs_dims.append(obs_dim)
            action_dims.append(action_space.n)
        
        self.args.obs_dim = obs_dims[0]  # Assume all agents have same obs dim
        self.args.action_dim = action_dims[0]  # Assume all agents have same action dim
        self.args.state_dim = np.sum(obs_dims)  # Global state as concatenation of all obs
        
        print("Number of training agents:", self.args.N)
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))

        # Create QMIX agent
        self.agent_n = QMIX(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir='runs/QMIX/QMIX_env_{}_number_{}_seed_{}'.format(
            self.env_name, self.number, self.seed))

        # Initialize epsilon for exploration
        self.epsilon = self.args.epsilon
        
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
        (self.line,) = self.ax.plot([], [], color='blue', label='QMIX')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Episode Reward')
        self.ax.set_title('MAgent2 Battle - QMIX')
        self.ax.legend(loc='lower right')
        self.fig.show()

        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)

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

            if self.replay_buffer.current_size >= self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)

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

        self.save_eval_csv()
        self.plot_eval_rewards()

    def save_eval_csv(self):
        csv_filename = './data_train/QMIX_env_{}_number_{}_seed_{}.csv'.format(
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
        plt.savefig('./data_train/QMIX_env_{}_number_{}_seed_{}_eval.png'.format(
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
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions for QMIX
        
        if self.args.use_rnn:
            self.agent_n.eval_Q_net.rnn_hidden = None
        
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
            
            # Pad observations if needed
            while len(obs_n) < self.args.N:
                obs_n.append(np.zeros(self.args.obs_dim))
            
            # Create global state and available actions
            s = np.concatenate(obs_n[:self.args.N])
            avail_a_n = np.ones((self.args.N, self.args.action_dim))  # All actions available in MAgent2
            
            # Get actions using current policy (no exploration during GIF generation)
            a_n = self.agent_n.choose_action(obs_n[:self.args.N], last_onehot_a_n, avail_a_n, epsilon=0)
            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]
            
            # Create action dictionary for all agents
            actions = {}
            
            # Add actions for active training agents
            action_idx = 0
            for agent in active_training_agents:
                if action_idx < len(a_n):
                    actions[agent] = a_n[action_idx]
                    action_idx += 1
            
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
        
        # Save GIF with 35 FPS
        if frames:
            gif_filename = f'./gifs/QMIX_env_{self.env_name}_step_{self.total_steps}.gif'
            # Calculate duration for 35 FPS: 1000ms / 35fps ≈ 28.57ms per frame
            duration_ms = int(1000 / 35)  # 28ms for 35 FPS
            
            frames[0].save(
                gif_filename,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,  # 28ms per frame for 35 FPS
                loop=0
            )
            print(f"GIF saved with 35 FPS: {gif_filename}")
        else:
            print("No frames captured for GIF")

    def run_episode_magent2(self, evaluate=False):
        episode_reward = 0
        observations, infos = self.env.reset(seed=self.seed)
        
        if self.args.use_rnn:
            self.agent_n.eval_Q_net.rnn_hidden = None
            
        episode_step = 0
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions for QMIX
        
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
            
            # Global state as concatenation of all observations
            s = np.concatenate(obs_n[:self.args.N])
            
            # Available actions (all actions are available in MAgent2)
            avail_a_n = np.ones((self.args.N, self.args.action_dim))
            
            # Choose actions
            epsilon = 0 if evaluate else self.epsilon
            a_n = self.agent_n.choose_action(obs_n[:self.args.N], last_onehot_a_n, avail_a_n, epsilon)
            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]
            
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
            r_total = 0
            for i, agent in enumerate(self.agent_list):
                if agent in rewards and i < len(active_training_agents):
                    r_total += rewards[agent]
            
            episode_reward += r_total

            # Store transition for training
            if not evaluate and len(active_training_agents) > 0:
                # Check if episode is done
                done = (all(terminations.values()) or all(truncations.values()) or 
                       episode_step + 1 >= self.args.episode_limit)
                
                # For QMIX, we need to determine if this is a "dead or win" situation
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False
                
                # Apply reward normalization if needed
                if self.args.use_reward_norm:
                    r_total = self.reward_norm(r_total)
                
                # Store transition
                self.replay_buffer.store_transition(episode_step, obs_n[:self.args.N], s, 
                                                  avail_a_n, last_onehot_a_n, a_n, r_total, dw)
                
                # Decay epsilon
                self.epsilon = (self.epsilon - self.args.epsilon_decay 
                              if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min 
                              else self.args.epsilon_min)

            episode_step += 1
            
            # Check if episode should end
            if (all(terminations.values()) or all(truncations.values()) or 
                episode_step >= self.args.episode_limit):
                break

        # Store last step for QMIX
        if not evaluate and len(obs_n) > 0:
            # Get final observations
            final_obs_n = []
            for agent in self.agent_list:
                if agent in observations:
                    obs = observations[agent]
                    if len(obs.shape) == 3:
                        obs = obs.flatten()
                    final_obs_n.append(obs)
                else:
                    final_obs_n.append(np.zeros(self.args.obs_dim))
            
            # Pad if needed
            while len(final_obs_n) < self.args.N:
                final_obs_n.append(np.zeros(self.args.obs_dim))
            
            final_s = np.concatenate(final_obs_n[:self.args.N])
            final_avail_a_n = np.ones((self.args.N, self.args.action_dim))
            
            self.replay_buffer.store_last_step(episode_step, final_obs_n[:self.args.N], 
                                             final_s, final_avail_a_n)

        return episode_reward, episode_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX in MAgent2 Battle environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=500, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Number of evaluations")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=50000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
    parser.add_argument("--use_RMS", type=bool, default=False, help="Whether to use RMS,if False, we will use Adam")
    parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
    parser.add_argument("--add_agent_id", type=bool, default=True, help="Whether to add agent id into the observation")
    parser.add_argument("--use_double_q", type=bool, default=True, help="Whether to use double q-learning")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
    parser.add_argument("--use_hard_update", type=bool, default=True, help="Whether to use hard update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
    parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")
    
    # MAgent2 specific arguments
    parser.add_argument("--map_size", type=int, default=15, help="Size of the battle map")
    parser.add_argument("--minimap_mode", type=bool, default=False, help="Use minimap observations")
    parser.add_argument("--extra_features", type=bool, default=False, help="Use extra features in observations")
    parser.add_argument("--num_agents", type=int, default=81, help="Number of agents to train")
    parser.add_argument("--train_team", type=str, default="red", choices=["red", "blue", "both"], 
                        help="Which team to train: red, blue, or both")

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

    runner = Runner_QMIX_MAgent2(args, env_name="battle", number=1, seed=0)
    runner.run()