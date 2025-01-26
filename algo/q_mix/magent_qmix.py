import yaml
from env import *
from win_rate import WinRate
from worker import RolloutWorker
from agent import Agents
from dqn_agents import dqnAgents
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
import logging

Log_Format = "%(levelname)s %(asctime)s - %(message)s"
import datetime

now = datetime.datetime.now()
fileName = "./log/" + str(now) + ".log"
logging.basicConfig(filename=fileName,
                    filemode="w",
                    format=Log_Format,
                    level=logging.INFO)

logger = logging.getLogger()

leftID, rightID = 0, 1


# 生成地图
def custom_generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""
    pos_left = []
    pos_right = []
    # left
    for i in range(1,26):
        for j in range(1,26):
            if j == 10 or j == 12:
                pos_left.append((i, j))
    env.add_agents(handles[leftID], method="custom", pos=pos_left)
    # print("left agemts num:", env.get_num(handles[leftID]))
    # right
    for m in range(1,26):
        for n in range(1,26):
            if n == 14 or n == 16:
                pos_right.append((m, n))
    env.add_agents(handles[rightID], method="custom", pos=pos_right)
    # print("right agemts num:", env.get_num(handles[rightID]))


def random_generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = 50  ##25v25

    # left

    env.add_agents(handles[leftID], method="random", n=50)

    # right

    env.add_agents(handles[rightID], method="random", n=50)


if __name__ == "__main__":
    with open('application.yaml', 'r') as file:
        config = yaml.safe_load(file)
    args = config['args']
    logger.info("初始化环境")
    magentEnv = MagentEnv(args)
    handles = magentEnv.handels
    magentEnv.env.set_render_dir('/home/lyf/qxx/MAgent-master/build/render')

    if args['rival'] == "qmix":
        group1_model_name = ['g1_qmix_net_params.pkl', 'g1_rnn_net_params.pkl']
        group2_model_name = ['g2_qmix_net_params.pkl', 'g2_rnn_net_params.pkl']
        logger.info("qmix vs qmix")
        agents_group1 = Agents(args, handles[0], models_names=group1_model_name, logger=logger)
        agents_group2 = Agents(args, handles[1], models_names=group2_model_name, logger=logger)
        agents_group2.load_model()
    elif args['rival'] == "dqn":
        group1_model_name = ['g1_qmix_net_params.pkl', 'g1_rnn_net_params.pkl']
        group2_model_name = "tfdqn"
        agents_group1 = Agents(args, handles[0], models_names=group1_model_name, logger=logger)
        agents_group2 = dqnAgents(args, handles[1], models_names=group2_model_name, env=magentEnv.env, logger=logger)
    worker = RolloutWorker(args, logger)
    worker_win_rate = WinRate(args, logger)
    buffer1 = ReplayBuffer(args)
    win_rate_record = []
    win_count = 0
    reward_count = 0
    avg_reward_record = []
    episode_rewards = []
    train_steps = 0
    save_path = args['result_dir'] + '/' + args['alg']
    max_win_rate = -1
    save_model_win_rate = False
    total_time_all = 0
    for epoch in range(args['n_epoch']):
        if epoch == 0:
            epoch = 1
        if epoch % 10 == 0 :
            logger.info("开始测试胜率：****************")
            print("开始测试胜率：****************")
            win_rate_count = 0
            reward_count = 0
            for i in range(10):
                magentEnv.env.reset()
                random_generate_map(magentEnv.env, args['map_size'], handles)
                # custom_generate_map(magentEnv.env, args['map_size'], handles)
                if_win, reward = worker_win_rate.generate_episode(magentEnv, epoch, agents_group1,
                                                                  agents_group2, epoch, evaluate=True)  # play
                reward_count += reward
                if if_win:
                    win_rate_count += 1
            logger.info("胜率：{}".format(win_rate_count / 10 * 100))

            print("胜率：{}%,平均奖励值：{}".format(win_rate_count / 10 * 100, reward_count / 10))
            avg_reward_record.append(reward_count / 10)
            print("*************************")
            ##保存模型
            win_rate_record.append(win_rate_count / 10)
            # if epoch ==50 or epoch== 100 or epoch == 180:
            #     agents_group1.policy.save_model(epoch)
            #     logger.info("save the best model !!!")

        episodes1 = []
        total_time = 0
        for episode_idx in range(args['n_episodes']):
            magentEnv.env.reset()
            random_generate_map(magentEnv.env, args['map_size'], handles)
            # custom_generate_map(magentEnv.env, args['map_size'], handles)
            episode, reward, if_win = worker.generate_episode(magentEnv,  agents_group1,
                                                              agents_group2, epoch, evaluate=False)  # play
            print(np.sum(reward))
            episodes1.append(episode)
        episode_batch1 = episodes1[0]
        buffer1.store_episode(episode_batch1)
        train_total_time = 0
        for train_step in range(args['train_steps']):  # train_steps:针对每一局进行采样，针对每次采样进行训练
            mini_batch1 = buffer1.sample(min(buffer1.current_size, args['batch_size']))
            train_time = agents_group1.train(mini_batch1, train_step)
            save_model_win_rate = False
            total_time += train_time
        total_time_all += total_time
        print("训练模型单局时间：{}".format(total_time))  # 改为19*20

        logger.info("训练模型总时间：{}".format(total_time_all))  # 改为19*20
        print("训练模型总时间：{}".format(total_time_all))

    print("胜率记录是：{}".format(win_rate_record))
    logger.info("胜率记录是：{}".format(win_rate_record))
    print("奖励记录是：{}".format(avg_reward_record))
    logger.info("奖励记录是：{}".format(avg_reward_record))
    plt.subplot(1, 2, 1)  # 1行2列，当前为第1个子图
    plt.plot(range(0, len(win_rate_record)), win_rate_record)
    plt.title('win rate')  # 设置第一个子图的标题
    plt.subplot(1, 2, 2)  # 1行2列，当前为第2个子图
    plt.plot(range(0, len(avg_reward_record)), avg_reward_record)
    plt.title('avg reward')  # 设置第一个子图的标题
    plt.show()
    plt.savefig("./plt/plt.png")
