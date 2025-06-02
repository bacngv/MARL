from magent2.environments import battle_v4

class MAgent2Wrapper:
    """
    Wrapper để làm cho MAgent2 Battle environment tương thích với PettingZoo MPE interface
    """
    def __init__(self, map_size=12, max_cycles=1000, minimap_mode=False, extra_features=False, render_mode=None):
        self.env = battle_v4.parallel_env(
            map_size=map_size,
            max_cycles=max_cycles,
            minimap_mode=minimap_mode,
            extra_features=extra_features,
            render_mode=render_mode,
            step_reward=0,
            dead_penalty=-0.05,
            attack_penalty=-0.001,
            attack_opponent_reward=0.2
        )
        
        # Lưu thông tin môi trường
        self.map_size = map_size
        self.max_cycles = max_cycles
        self.minimap_mode = minimap_mode
        self.extra_features = extra_features
        
        # Reset để lấy thông tin agents
        observations, infos = self.env.reset()
        self.agents = list(observations.keys())
        
        # Chia agents thành 2 teams
        self.red_agents = [agent for agent in self.agents if agent.startswith('red_')]
        self.blue_agents = [agent for agent in self.agents if agent.startswith('blue_')]
        
        print(f"Red agents: {len(self.red_agents)}")
        print(f"Blue agents: {len(self.blue_agents)}")
        
    def reset(self, seed=None):
        """Reset environment"""
        observations, infos = self.env.reset(seed=seed)
        return observations, infos
    
    def step(self, actions):
        """Step environment"""
        return self.env.step(actions)
    
    def render(self):
        """Render environment"""
        return self.env.render()
    
    def close(self):
        """Close environment"""
        self.env.close()
    
    def observation_space(self, agent):
        """Get observation space for agent"""
        return self.env.observation_space(agent)
    
    def action_space(self, agent):
        """Get action space for agent"""
        return self.env.action_space(agent)
    
    @property
    def possible_agents(self):
        """Get possible agents"""
        return self.env.possible_agents
