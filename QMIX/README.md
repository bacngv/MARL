# QMIX and VDN in StarCraft II environment
This is a concise Pytorch implementation of QMIX and VDN in MAgent2 environment.<br />

## How to use my code?
You can dircetly run 'QMIX_main.py' in your own IDE.<br />
If you want to use QMIX, you can set the paraemeter 'algorithm' = 'QMIX';<br />
If you want to use VDN, you can set the paraemeter 'algorithm' = 'VDN'.<br />

## Trainning environments
- Check out the [PIPELINE](https://colab.research.google.com/drive/1MKJx-ihFlReny8CL6BSgNMQduuQyfMcH)
- We train our QMIX in 'battle_v4' in MAgent2 environment.<br />

## Requirements
```
pip install -r requirements.txt
```

## Reference
[1] Rashid T, Samvelyan M, Schroeder C, et al. Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning[C]//International Conference on Machine Learning. PMLR, 2018: 4295-4304.<br />
[2] Sunehag P, Lever G, Gruslys A, et al. Value-decomposition networks for cooperative multi-agent learning[J]. arXiv preprint arXiv:1706.05296, 2017.<br />
[3] [EPyMARL](https://github.com/uoe-agents/epymarl).<br />
[4] https://github.com/starry-sky6688/StarCraft.<br />
