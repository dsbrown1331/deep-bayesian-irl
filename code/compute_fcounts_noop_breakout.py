#make sure it uses the custom baselines package
import sys
sys.path.insert(0,'./baselines/')

import argparse
# coding: utf-8

# Take as input a compressed pretrained network or run T_REX before hand
# Then run MCMC and save posterior chain


import pickle
import copy
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *
from StrippedNet import EmbeddingNet
from baselines.common.trex_utils import preprocess


def generate_mean_map_noop_demos(env):


    #add no-op demos
    done = False
    traj = []
    gt_rewards = []
    r = 0

    ob = env.reset()
    steps = 0
    acc_reward = 0
    while steps < 7000:
        action = 0#agent.act(ob, r, done)
        ob, r, done, _ = env.step(action)
        ob_processed = preprocess(ob, env_name)
        #ob_processed = ob_processed[0] #get rid of first dimension ob.shape = (1,84,84,4)
        traj.append(ob_processed)

        gt_rewards.append(r[0])
        steps += 1
        acc_reward += r[0]
        if done:
            print("checkpoint: {}, steps: {}, return: {}".format("noop", steps,acc_reward))
            break
    print("noop traj length", len(traj))


    return traj, acc_reward, gt_rewards


def generate_feature_counts(demos, reward_net):
    feature_cnts = torch.zeros(len(demos), reward_net.fc2.in_features) #no bias
    for i in range(len(demos)):
        traj = np.array(demos[i])
        traj = torch.from_numpy(traj).float().to(device)
        for s in traj:
            print(reward_net.state_feature(s))
        #print(len(trajectory))
        feature_cnts[i,:] = reward_net.state_features(traj).squeeze().float().to(device)
    return feature_cnts.to(device)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='breakout ', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--seed', default=1234, help="random seed for experiments")
    parser.add_argument('--encoding_dims', default=64, type=int, help='number of dims to encode to')
    parser.add_argument('--pretrained_network', help='path to directory of pretrained network weights to form \phi(s) using all but last layer')



    args = parser.parse_args()


    env_name = args.env_name
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

    env_type = "atari"
    print(env_type)
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    stochastic = True


    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })


    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)


    demonstrations, learning_returns, learning_rewards = generate_mean_map_noop_demos(env)


    # Now we download a pretrained network to form \phi(s) the state features where the reward is now w^T \phi(s)
    print("loading policy", args.pretrained_network)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = EmbeddingNet(args.encoding_dims)
    reward_net.load_state_dict(torch.load(args.pretrained_network, map_location=device))
    #reinitialize last layer
    num_features = reward_net.fc2.in_features

    print("reward is linear combination of ", num_features, "features")
    reward_net.fc2 = nn.Linear(num_features, 1, bias=False) #last layer just outputs the scalar reward = w^T \phi(s)
    reward_net.to(device)
    #freeze all weights so there are no gradients (we'll manually update the last layer via proposals so no grads required)
    for param in reward_net.parameters():
        param.requires_grad = False

    #get num_demos by num_features + 1 (bias) numpy array with (un-discounted) feature counts from pretrained network
    demo_cnts = generate_feature_counts([demonstrations], reward_net)
    print("demo counts")
    print(demo_cnts)
