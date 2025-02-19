import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from moviepy.editor import ImageSequenceClip
import threading
import pandas as pd


class Color:
    INFO = '\033[1;34m{}\033[0m'
    WARNING = '\033[1;33m{}\033[0m'
    ERROR = '\033[1;31m{}\033[0m'
    
class Runner(object):
    def __init__(self, env, handles, max_steps, models,
                play_handle, render_every=None, save_every=None, tau=0.001, 
                log_name=None, log_dir=None, model_dir=None, render_dir=None, 
                train=False, cuda=True):
        self.env = env
        self.models = models
        self.max_steps = max_steps
        self.handles = handles
        self.render_every = render_every
        self.save_every = save_every
        self.play = play_handle
        self.model_dir = model_dir
        self.render_dir = render_dir
        self.train = train
        self.tau = tau
        self.cuda = cuda
        self.log_dir = log_dir
        self._create_directories()
        
    def _create_directories(self):
        """Create all necessary directories for logs, models, and renders."""
        directories = [self.log_dir, self.model_dir, self.render_dir]
        for directory in directories:
            if directory is not None:
                os.makedirs(directory, exist_ok=True)
                print(f"[INFO] Created directory: {directory}")

    def save_to_csv(self, round_losses, round_rewards, iteration):
        """Save round losses and rewards to a CSV file."""
        if self.log_dir is None:
            print("[WARNING] Log directory not specified, skipping CSV save")
            return
        data = {
            'Round': iteration,
            'Loss': round_losses[0][-1] if round_losses[0] else None,
            'Main Reward': round_rewards[0][-1] if round_rewards[0] else None,
            'Opponent Reward': round_rewards[1][-1] if round_rewards[1] else None,
        }
        
        df = pd.DataFrame([data])
        csv_path = os.path.join(self.log_dir, 'training_metrics.csv')
        
        try:
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, mode='w', header=True, index=False)
            print(f"[INFO] Successfully saved metrics to {csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save metrics to CSV: {str(e)}")
    
    def is_self_play(self):
        """Check if we're in self-play mode"""
        return hasattr(self.models[1], 'save')
    
    def soft_update(self):
        """Perform soft update from main model to opponent model in self-play"""
        if not self.is_self_play():
            return
            
        l_vars, r_vars = self.models[0].get_all_params(), self.models[1].get_all_params()
        for l_var, r_var in zip(l_vars, r_vars):
            r_var.detach().copy_((1. - self.tau) * l_var + self.tau * r_var)
            
    def run(self, variant_eps, iteration, win_cnt=None):
        info = {'main': None, 'opponent': None}
        info['main'] = {'ave_agent_reward': 0., 'total_reward': 0., 'kill': 0.}
        info['opponent'] = {'ave_agent_reward': 0., 'total_reward': 0., 'kill': 0.}
        
        max_nums, nums, mean_rewards, total_rewards, render_list = self.play(
            env=self.env, 
            n_round=iteration, 
            handles=self.handles,
            models=self.models, 
            print_every=50, 
            eps=variant_eps, 
            render=(iteration + 1) % self.render_every == 0 if self.render_every > 0 else False, 
            train=self.train, 
            cuda=self.cuda
        )
        
        for i, tag in enumerate(['main', 'opponent']):
            info[tag]['total_reward'] = total_rewards[i]
            info[tag]['kill'] = max_nums[i] - nums[1 - i]
            info[tag]['ave_agent_reward'] = mean_rewards[i]
            
        if self.train:
            print('\n[INFO] Main: {} \nOpponent: {}'.format(info['main'], info['opponent']))
            
            if info['main']['total_reward'] > info['opponent']['total_reward']:
                if self.is_self_play():
                    print('[INFO] Self-play mode - performing soft update...')
                    self.soft_update()
                    print('[INFO] Soft update completed')
                
                print('[INFO] Saving main model...')
                self.models[0].save(self.model_dir + '-main', iteration)
                if self.is_self_play():
                    print('[INFO] Saving opponent model...')
                    self.models[1].save(self.model_dir + '-opponent', iteration)
        else:
            if win_cnt is not None:
                if info['main']['kill'] > info['opponent']['kill']:
                    win_cnt['main'] += 1
                elif info['main']['kill'] < info['opponent']['kill']:
                    win_cnt['opponent'] += 1
                else:
                    win_cnt['main'] += 1
                    win_cnt['opponent'] += 1
        
        if len(render_list) > 0:
            print('[*] Saving Render')
            clip = ImageSequenceClip(render_list, fps=35)
            clip.write_gif('{}/replay_{}.gif'.format(self.render_dir, iteration+1), fps=20, verbose=False)
            print('[*] Saved Render')