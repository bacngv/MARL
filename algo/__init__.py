from . import q_learning
from . import ac
from . import qmix
from . import mappo

IQL = q_learning.DQN
MFQ = q_learning.MFQ
AC = ac.ActorCritic
MFAC = ac.MFAC
QMIX = qmix.QMIX
MAPPO = mappo.MAPPOPolicy


def spawn_ai(algo_name, env, handle, human_name, max_steps, cuda=True):
    if algo_name == 'mfq':
        model = MFQ(env, human_name, handle, max_steps, memory_size=80000)
    elif algo_name == 'iql':
        model = IQL(env, human_name, handle, max_steps, memory_size=80000)
    elif algo_name == 'ac':
        model = AC(env, human_name, handle, use_cuda=cuda)
    elif algo_name == 'mfac':
        model = MFAC(env, human_name, handle, use_cuda=cuda)
    elif algo_name == 'mappo':
        model = MAPPO(env, human_name, handle, use_cuda=cuda)
    if cuda:
        model = model.cuda()
    return model