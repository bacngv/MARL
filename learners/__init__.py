from . import q_learning
from . import ac
from . import mappo
from . import ppo
from . import vdn

IQL = q_learning.IQL
MFQ = q_learning.MFQ
AC = ac.ActorCritic
MFAC = ac.MFAC
MAPPO = mappo.MAPPO
PPO = ppo.PPO
VDN = vdn.VDN
QMIX = q_learning.QMix

def spawn_ai(algo_name, env, handle, human_name, max_steps, cuda=True):
    num = env.unwrapped.env.get_num(handle)
    if algo_name == 'mfq':
        model = MFQ(env, human_name, handle, max_steps, memory_size=40000)
    elif algo_name == 'iql':
        model = IQL(env, human_name, handle, max_steps, memory_size=40000)
    elif algo_name == 'ppo':
        model = PPO(env, human_name, handle, max_steps, memory_size=40000)
    elif algo_name == 'ac':
        model = AC(env, human_name, handle, use_cuda=cuda)
    elif algo_name == 'mfac':
        model = MFAC(env, human_name, handle, use_cuda=cuda)
    elif algo_name == 'mappo':
        model = MAPPO(env, human_name, handle,num_agents=num, sub_len=400, memory_size=40000)
    elif algo_name == 'vdn':
        model = VDN(env, human_name, handle,num_agents=num, sub_len=400, memory_size=40000)
    elif algo_name == 'qmix':
        model = QMIX(env, human_name, handle,num_agents=num, sub_len=400, memory_size=40000)
    if cuda:
        model = model.cuda()
    return model