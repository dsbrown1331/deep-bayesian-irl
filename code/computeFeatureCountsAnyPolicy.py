import os
import sys
import pickle
import gym
import time
import numpy as np
import random
import torch
from run_test import *
#import matplotlib.pylab as plt
import argparse
import nnet
from baselines.common.trex_utils import preprocess
import utils

def get_policy_feature_counts(env_name, checkpointpath, feature_net, num_rollouts, fixed_horizon):
    if env_name == "spaceinvaders":
        env_id = "SpaceInvaders"
    elif env_name == "mspacman":
        env_id = "MsPacman"
    elif env_name == "videopinball":
        env_id = "VideoPinball"
    elif env_name == "beamrider":
        env_id = "BeamRider"
    elif env_name == "montezumarevenge":
        env_id = "MontezumaRevenge"
    else:
        env_id = env_name[0].upper() + env_name[1:]

    if fixed_horizon:
        env_id += "NoFrameskipFixedHorizon-v0"
    else:
        env_id += "NoFrameskip-v4"

    env_type = "atari"

    stochastic = True

    #env id, env type, num envs, and seed
    env = make_vec_env(env_id, 'atari', 1, 0,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })



    env = VecFrameStack(env, 4)


    agent = PPO2Agent(env, env_type, stochastic)  #defaults to stochastic = False (deterministic policy)
    #agent = RandomAgent(env.action_space)

    learning_returns = []

    print(checkpointpath)

    agent.load(checkpointpath)
    episode_count = num_rollouts

    f_counts = np.zeros(feature_net.fc2.in_features + 1)

    for i in range(episode_count):
        done = False
        traj = []
        r = 0

        ob = env.reset()
        #traj.append(ob)
        #print(ob.shape)
        steps = 0
        acc_reward = 0
        while True:
            action = agent.act(ob, r, done)
            #print(action)
            ob, r, done, _ = env.step(action)
            ob_processed = preprocess(ob, env_name)
            #print(ob_processed.shape)
            phi_s = torch.cat((feature_net.state_feature(torch.from_numpy(ob_processed).float().to(device)).cpu().squeeze(), torch.tensor([1.]))).numpy()
            #print(phi_s.shape)
            f_counts += phi_s
            steps += 1
            #print(steps)
            acc_reward += r[0]
            if done:
                print("steps: {}, return: {}".format(steps,acc_reward))
                break

        learning_returns.append(acc_reward)



    env.close()
    #tf.reset_default_graph()

    ave_fcounts = f_counts/episode_count

    return learning_returns, ave_fcounts

    #return([np.max(learning_returns), np.min(learning_returns), np.mean(learning_returns)])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=1234, help="random seed for experiments")
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--checkpointpath', default='', help='path to checkpoint to run eval on')
    parser.add_argument('--pretrained_network', help='path to pretrained network weights to form \phi(s) using all but last layer')
    parser.add_argument('--num_rollouts', type=int, help='number of rollouts to compute feature counts')
    parser.add_argument('--output_id', default='', help='unique id for output file name')
    parser.add_argument('--fixed_horizon', action='store_true')


    args = parser.parse_args()
    env_name = args.env_name
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_net = nnet.Net()
    feature_net.load_state_dict(torch.load(args.pretrained_network))
    feature_net.to(device)

    checkpointpath = args.checkpointpath
    print("*"*10)
    print(env_name)
    print("*"*10)
    returns, ave_feature_counts = get_policy_feature_counts(env_name, checkpointpath, feature_net, args.num_rollouts, args.fixed_horizon)
    print("returns", returns)
    print("feature counts", ave_feature_counts)
    writer = open("../policies/" + env_name + "_" + args.output_id + "_fcounts.txt", 'w')
    utils.write_line(ave_feature_counts, writer)
    utils.write_line(returns, writer, newline=False)
    writer.close()
