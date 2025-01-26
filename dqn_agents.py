from magent.builtin.tf_model import DeepQNetwork
import magent2
class dqnAgents:
    def __init__(self, args, handel, models_names, env, logger):
        self.model_name = "dqn"
        self.num_actions = args['num_actions']
        self.num_agents = args['num_agents']
        self.state_space = args['state_space']
        self.obs_space = args['obs_space']
        self.args = args
        self.handel = handel
        self.models_names = models_names
        self.env = env
        self.logger = logger

        batch_size = 256
        target_update = 1200
        train_freq = 5
        model = DeepQNetwork(self.env, self.handel, 'battle-r',
                             batch_size=batch_size,
                             learning_rate=3e-4,
                             memory_size=2 ** 21, target_update=target_update,
                             train_freq=train_freq)

        # model.save('../../save_model', 15)
        model.load('../../save_model', 15)
        logger.info("loading model!!!!")
        self.policy = model

    def choose_action(self, obs, k, evaluate):
        ids = self.env.get_agent_id(self.handel)
        eps = magent.utility.piecewise_decay(k, [0, 700, 1400], [1, 0.2, 0.05])  # 开启效果最好
        if evaluate:
            eps = 0
        eps=1
        action = self.policy.infer_action(obs, ids, 'e_greedy', eps)
        return action
