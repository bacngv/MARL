import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from . import base
from . import tools

class MAPPO(base.ValueNet):
    def __init__(self, env, name, handle, num_agents=30, use_cuda=False, memory_size=2**10, batch_size=64,
                 update_every=5, use_mf=False, learning_rate=0.0001, clip_param=0.2,
                 value_coef=0.5, entropy_coef=0.01, gamma=0.95, gae_lambda=0.95, tau=0.005):
        super().__init__(env, name, handle, update_every=update_every, use_mf=use_mf,
                        learning_rate=learning_rate, gamma=gamma)
        
        self.num_agents = num_agents
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.tau = tau
        self.use_cuda = use_cuda
        
        # Modified replay buffer to handle multiple agents
        self.replay_buffer = tools.MemoryGroup(self.view_space, self.feature_space,
                                             self.num_actions, memory_size, batch_size, 
                                             sub_len=self.num_agents)
        
        self.actor = nn.ModuleList([self._construct_actor() for _ in range(num_agents)])
        self.critic = self._construct_critic()  # Centralized critic
        self.target_actor = nn.ModuleList([self._construct_actor() for _ in range(num_agents)])
        self.target_critic = self._construct_critic()  # Centralized target critic
        
        # Initialize target networks
        for i in range(num_agents):
            self.target_actor[i].load_state_dict(self.actor[i].state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Move models to CUDA if needed
        if self.use_cuda:
            for i in range(num_agents):
                self.actor[i] = self.actor[i].cuda()
                self.target_actor[i] = self.target_actor[i].cuda()
            self.critic = self.critic.cuda()
            self.target_critic = self.target_critic.cuda()
        
        # Separate optimizers for each actor and one for critic
        self.actor_optims = [torch.optim.Adam(self.get_params(actor), lr=learning_rate) 
                            for actor in self.actor]
        self.critic_optim = torch.optim.Adam(self.get_params(self.critic), lr=learning_rate)


    def _construct_actor(self):
        temp_dict = nn.ModuleDict()
        temp_dict['conv1'] = nn.Conv2d(in_channels=self.view_space[2], out_channels=32, kernel_size=3)
        temp_dict['conv2'] = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        temp_dict['obs_linear'] = nn.Linear(self.get_flatten_dim(temp_dict), 256)
        temp_dict['emb_linear'] = nn.Linear(self.feature_space, 32)
        if self.use_mf:
            temp_dict['prob_emb_linear'] = nn.Sequential(
                nn.Linear(self.num_actions, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
        temp_dict['final_linear'] = nn.Sequential(
            nn.Linear(320 if self.use_mf else 288, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )
        return temp_dict

    def _construct_critic(self):
        temp_dict = nn.ModuleDict()
        
        # Process each agent's observations separately first
        temp_dict['conv1_list'] = nn.ModuleList([
            nn.Conv2d(in_channels=self.view_space[2], out_channels=32, kernel_size=3)
            for _ in range(self.num_agents)
        ])
        temp_dict['conv2_list'] = nn.ModuleList([
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
            for _ in range(self.num_agents)
        ])
        
        # Calculate flattened dimension for one agent
        sample_obs = torch.zeros(1, self.view_space[2], self.view_space[0], self.view_space[1])
        conv1_out = temp_dict['conv1_list'][0](sample_obs)
        conv2_out = temp_dict['conv2_list'][0](conv1_out)
        single_agent_dim = conv2_out.flatten().size()[0]
        
        # Linear layers
        temp_dict['obs_linear'] = nn.Linear(single_agent_dim * self.num_agents, 256)
        temp_dict['emb_linear'] = nn.Linear(self.feature_space * self.num_agents, 32)
        
        if self.use_mf:
            temp_dict['prob_emb_linear'] = nn.Sequential(
                nn.Linear(self.num_actions * self.num_agents, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
            
        temp_dict['final_linear'] = nn.Sequential(
            nn.Linear(320 if self.use_mf else 288, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_agents)
        )
        return temp_dict



    def act(self, obs=None, feature=None, prob=None, eps=None):
        """
        Get action from the agent.
        
        Args:
            obs: observation tensor
            feature: feature tensor
            prob: probability tensor
            eps: epsilon for exploration
            
        Returns:
            numpy array of actions
        """
        with torch.no_grad():
            # Process single observation for the current agent
            a_h = F.relu(self.actor[0]['conv2'](
                F.relu(self.actor[0]['conv1'](obs)))).flatten(start_dim=1)
            a_h = torch.cat([self.actor[0]['obs_linear'](a_h), 
                           self.actor[0]['emb_linear'](feature)], -1)
            
            if self.use_mf and prob is not None:
                a_h = torch.cat([a_h, self.actor[0]['prob_emb_linear'](prob)], -1)
                
            logits = self.actor[0]['final_linear'](a_h)
            probs = F.softmax(logits, dim=-1)
            
            # Apply epsilon-greedy if eps is provided
            if eps is not None:
                random_action = torch.randint(0, self.num_actions, probs.shape[:-1])
                random_mask = torch.rand(probs.shape[:-1]) < eps
                if self.use_cuda:
                    random_action = random_action.cuda()
                    random_mask = random_mask.cuda()
                action = torch.where(random_mask, random_action, probs.argmax(dim=-1))
            else:
                dist = Categorical(probs)
                action = dist.sample()
            
            # Return action as numpy array
            return action.cpu().numpy()

    def get_action_and_value(self, obs_n, feature_n, prob_n=None, action_n=None):
        """Get actions, log probabilities, entropies and values for all agents"""
        batch_size = obs_n[0].shape[0]
        actions = []
        log_probs = []
        entropies = []
        
        # Get actions from each agent's actor network
        for i in range(self.num_agents):
            # Process observations through conv layers
            a_h = F.relu(self.actor[i]['conv2'](
                F.relu(self.actor[i]['conv1'](obs_n[i])))).flatten(start_dim=1)
                
            # Process embeddings
            emb = self.actor[i]['emb_linear'](feature_n[i])
            
            # Ensure both tensors have the same dimensions before concatenating
            if len(a_h.shape) != len(emb.shape):
                if len(a_h.shape) < len(emb.shape):
                    a_h = a_h.unsqueeze(-1)
                else:
                    emb = emb.unsqueeze(-1)
                    
            # Concatenate processed observations and embeddings
            a_h = torch.cat([a_h, emb], dim=-1)
            
            if self.use_mf and prob_n is not None:
                prob_emb = self.actor[i]['prob_emb_linear'](prob_n[i])
                # Ensure prob_emb has same dimensions
                if len(prob_emb.shape) != len(a_h.shape):
                    if len(prob_emb.shape) < len(a_h.shape):
                        prob_emb = prob_emb.unsqueeze(-1)
                    else:
                        a_h = a_h.unsqueeze(-1)
                a_h = torch.cat([a_h, prob_emb], dim=-1)
            
            # Get action logits and probabilities
            logits = self.actor[i]['final_linear'](a_h)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            if action_n is None:
                action = dist.sample()
            else:
                action = action_n[i]
            
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
        
        # Get centralized value
        values = self.get_value(obs_n, feature_n, prob_n)
        
        return actions, log_probs, entropies, values

    def get_value(self, obs_n, feature_n, prob_n=None):
        """Get centralized value estimate for all agents"""
        batch_size = obs_n[0].shape[0]
        
        # Process each agent's observations separately
        processed_obs = []
        for i in range(self.num_agents):
            x = F.relu(self.critic['conv1_list'][i](obs_n[i]))
            x = F.relu(self.critic['conv2_list'][i](x))
            processed_obs.append(x.flatten(start_dim=1))
        
        # Concatenate all processed observations
        concat_obs = torch.cat(processed_obs, dim=1)
        concat_feature = torch.cat(feature_n, dim=1)
        
        # Process through critic network
        c_h = torch.cat([
            self.critic['obs_linear'](concat_obs),
            self.critic['emb_linear'](concat_feature)
        ], dim=-1)
        
        if self.use_mf and prob_n is not None:
            concat_prob = torch.cat(prob_n, dim=1)
            prob_emb = self.critic['prob_emb_linear'](concat_prob)
            c_h = torch.cat([c_h, prob_emb], dim=-1)
        
        values = self.critic['final_linear'](c_h)
        
        return values

    def update(self):
        """Soft update target networks using tau parameter"""
        # Update all actors
        for i in range(self.num_agents):
            for param, target_param in zip(self.actor[i].parameters(), 
                                         self.target_actor[i].parameters()):
                target_param.data.copy_(self.tau * param.data + 
                                      (1 - self.tau) * target_param.data)
        
        # Update critic
        for param, target_param in zip(self.critic.parameters(), 
                                     self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                  (1 - self.tau) * target_param.data)

    def train(self, cuda):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()
        total_loss = 0
        actor_losses = [0] * self.num_agents
        critic_loss = 0
        entropy_losses = [0] * self.num_agents
        
        for i in range(batch_num):
            # Sample batch for each agent
            obs_n, feat_n, obs_next_n, feat_next_n, dones_n, rewards_n, acts_n, masks_n = \
                self.replay_buffer.sample()
            
            # Process tensors for each agent
            obs_n = [torch.FloatTensor(obs) for obs in obs_n]
            obs_next_n = [torch.FloatTensor(obs_next) for obs_next in obs_next_n]
            
            # Check and handle observation dimensions
            for idx in range(len(obs_n)):
                if len(obs_n[idx].shape) == 3:  # If missing batch dimension
                    obs_n[idx] = obs_n[idx].unsqueeze(0)
                if len(obs_next_n[idx].shape) == 3:
                    obs_next_n[idx] = obs_next_n[idx].unsqueeze(0)
                
                # Now permute with proper dimensions
                obs_n[idx] = obs_n[idx].permute(0, 3, 1, 2)
                obs_next_n[idx] = obs_next_n[idx].permute(0, 3, 1, 2)
            
            feat_n = [torch.FloatTensor(feat) for feat in feat_n]
            feat_next_n = [torch.FloatTensor(feat_next) for feat_next in feat_next_n]
            acts_n = [torch.LongTensor(acts) for acts in acts_n]
            
            # Handle rewards - ensure they're in the correct format
            if isinstance(rewards_n, (list, tuple)):
                # If rewards_n is already a list of rewards
                rewards_n = [torch.FloatTensor([r] if isinstance(r, (float, np.float32)) else r) 
                            for r in rewards_n]
            else:
                # If rewards_n is a single value, convert to list of tensors
                rewards_n = [torch.FloatTensor([rewards_n])] * self.num_agents
                
            # Handle dones in a similar way
            if isinstance(dones_n, (list, tuple)):
                dones_n = [torch.FloatTensor([d] if isinstance(d, (float, np.float32)) else d) 
                          for d in dones_n]
            else:
                dones_n = [torch.FloatTensor([dones_n])] * self.num_agents
                
            # Handle masks in a similar way
            if isinstance(masks_n, (list, tuple)):
                masks_n = [torch.FloatTensor([m] if isinstance(m, (float, np.float32)) else m) 
                          for m in masks_n]
            else:
                masks_n = [torch.FloatTensor([masks_n])] * self.num_agents
            
            if cuda:
                obs_n = [obs.cuda() for obs in obs_n]
                obs_next_n = [obs_next.cuda() for obs_next in obs_next_n]
                feat_n = [feat.cuda() for feat in feat_n]
                feat_next_n = [feat_next.cuda() for feat_next in feat_next_n]
                acts_n = [acts.cuda() for acts in acts_n]
                rewards_n = [rewards.cuda() for rewards in rewards_n]
                dones_n = [dones.cuda() for dones in dones_n]
                masks_n = [masks.cuda() for masks in masks_n]
            
            # Get new action probabilities and values
            _, new_log_probs_n, entropies_n, values = self.get_action_and_value(
                obs_n, feat_n, action_n=acts_n)
            
            # Calculate advantages for each agent
            advantages_n = [torch.zeros_like(rewards) for rewards in rewards_n]
            lastgaelam_n = [0] * self.num_agents
            
            with torch.no_grad():
                next_values = self.get_value(obs_next_n, feat_next_n)
                
            # Calculate GAE for each agent
            for agent_idx in range(self.num_agents):
                for t in reversed(range(len(rewards_n[agent_idx]))):
                    if t == len(rewards_n[agent_idx]) - 1:
                        nextnonterminal = 1.0 - dones_n[agent_idx][t]
                        nextvalues = next_values[t][agent_idx]
                    else:
                        nextnonterminal = 1.0 - dones_n[agent_idx][t]
                        nextvalues = values[t + 1][agent_idx]
                    
                    delta = rewards_n[agent_idx][t] + self.gamma * nextvalues * \
                          nextnonterminal - values[t][agent_idx]
                    advantages_n[agent_idx][t] = lastgaelam_n[agent_idx] = delta + \
                        self.gamma * self.gae_lambda * nextnonterminal * lastgaelam_n[agent_idx]
            
            # Calculate returns and normalize advantages
            returns_n = [advantages + values[:, i] for i, advantages in 
                        enumerate(advantages_n)]
            advantages_n = [(advantages - advantages.mean()) / (advantages.std() + 1e-8)
                          for advantages in advantages_n]
            
            # Calculate losses for each agent
            policy_losses = []
            entropy_losses = []
            
            for agent_idx in range(self.num_agents):
                ratio = torch.exp(new_log_probs_n[agent_idx])
                surr1 = ratio * advantages_n[agent_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * \
                        advantages_n[agent_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropies_n[agent_idx].mean()
                
                policy_losses.append(policy_loss)
                entropy_losses.append(entropy_loss)
            
            # Value loss (centralized critic)
            value_loss = sum(F.mse_loss(values[:, i], returns_n[i]) 
                          for i in range(self.num_agents))
            
            # Update networks
            for agent_idx in range(self.num_agents):
                self.actor_optims[agent_idx].zero_grad()
                loss = policy_losses[agent_idx] + self.entropy_coef * entropy_losses[agent_idx]
                loss.backward(retain_graph=True)
                self.actor_optims[agent_idx].step()
                
                actor_losses[agent_idx] += policy_losses[agent_idx].item()
                entropy_losses[agent_idx] += entropy_losses[agent_idx].item()
            
            self.critic_optim.zero_grad()
            (self.value_coef * value_loss).backward()
            self.critic_optim.step()
            
            critic_loss += value_loss.item()
            
            if i % 50 == 0:
                print(f'[*] Batch {i}/{batch_num}')
                for agent_idx in range(self.num_agents):
                    print(f'Agent {agent_idx} - Policy Loss: {policy_losses[agent_idx].item():.4f}, '
                          f'Entropy Loss: {entropy_losses[agent_idx].item():.4f}')
                print(f'Critic Loss: {value_loss.item():.4f}\n')

    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        
        # Save actors
        for i in range(self.num_agents):
            actor_path = os.path.join(dir_path, f"mappo_actor_{i}_{step}")
            torch.save(self.actor[i].state_dict(), actor_path)
            
        # Save critic
        critic_path = os.path.join(dir_path, f"mappo_critic_{step}")
        torch.save(self.critic.state_dict(), critic_path)
        print("[*] Model saved")
        
    def load(self, dir_path, step=0):
        # Load actors
        for i in range(self.num_agents):
            actor_path = os.path.join(dir_path, f"mappo_actor_{i}_{step}")
            self.actor[i].load_state_dict(torch.load(actor_path))
            
        # Load critic
        critic_path = os.path.join(dir_path, f"mappo_critic_{step}")
        self.critic.load_state_dict(torch.load(critic_path))
        
        # Update target networks
        for i in range(self.num_agents):
            self.target_actor[i].load_state_dict(self.actor[i].state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        print("[*] Loaded model")
        
    def get_all_params(self):
        """
        Get all parameters from both actor and critic networks for self-play.
        
        Returns:
            list: List of all parameters from both networks
        """
        params = []
        # Get parameters from all actors
        for actor in self.actor:
            for k, v in actor.items():
                params.extend(list(v.parameters()))
        # Get parameters from critic
        for k, v in self.critic.items():
            params.extend(list(v.parameters()))
        return params
    
    def get_params(self, network_dict):
        """Helper method to get parameters from a network dictionary"""
        params = []
        for k, v in network_dict.items():
            params.extend(list(v.parameters()))
        return params
    
    def flush_buffer(self, **kwargs):
        """
        Push data to the replay buffer.
        
        Args:
            **kwargs: Dictionary containing the following keys for each agent:
                - obs_n: list of observations for each agent
                - feature_n: list of additional features for each agent
                - acts_n: list of actions taken by each agent
                - rewards_n: list of rewards received by each agent
                - alives_n: list of alive status for each agent
                - obs_next_n: list of next observations for each agent
                - feature_next_n: list of next additional features for each agent
        """
        self.replay_buffer.push(**kwargs)