# MAPPO in MAgent2 environment
This is a concise Pytorch implementation of MAPPO in MAgent2 environment.<br />
This code only works in the environments where all agents are homogenous, such as 'battlev4' in MAgent2. Here, all agents have the same dimension of observation space and action space.<br />

## How to use my code?  
You can dircetly run 'MAPPO_main.py' in your own IDE.<br />

## Trainning environments
- Check out the [PIPELINE](https://colab.research.google.com/drive/1veHUQ3242LK_JJA9LmIPq3T2U95ANHTF)
- We train our MAPPO in 'battle_v4' in MAgent2 environment.<br />

## Requirements
```
pip install -r requirements.txt
```

## Some details
Because the MAgent2 environment is is relatively simple, we do not use RNN in 'actor' and 'critic' which can result in the better performence according to our experimental results.<br />
However, we also provide the implementation of using RNN. You can set 'use_rnn'=True in the hyperparameters setting, if you want to use RNN.<br />

## Reference
[1] Yu C, Velu A, Vinitsky E, et al. The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games[J]. arXiv preprint arXiv:2103.01955, 2021.<br />
[2] [Official implementation of MAPPO](https://github.com/marlbenchmark/on-policy)<br />
[3] [EPyMARL](https://github.com/uoe-agents/epymarl)
