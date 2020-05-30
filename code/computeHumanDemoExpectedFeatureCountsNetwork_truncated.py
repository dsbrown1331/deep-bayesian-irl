### This code will take in any pretrained network and compute the expected feature counts via Monte Carlo sampling according to the last
### layer of the pretrained network


#will only run for 2000 steps.

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
from StrippedNet import EmbeddingNet
from baselines.common.trex_utils import preprocess
import utils




def get_demo_feature_counts(env_name, trajectory, feature_net, max_length):
    learning_returns = []
    fcount_rollouts = [] #keep track of the feature counts for each rollout
    num_steps = []

    f_counts = np.zeros(feature_net.fc2.in_features)

    steps = 0

    for i in range(min(max_length, len(trajectory))):
        ob = trajectory[i]
        steps += 1
        done = False
        traj = []
        r = 0
        ob_processed = preprocess(ob, env_name)
        phi_s = feature_net.state_feature(torch.from_numpy(ob_processed).float().to(device)).cpu().squeeze().numpy()
        f_counts += phi_s

    ave_fcounts = f_counts
    fcount_rollouts.append(ave_fcounts)
    #print('ave', ave_fcounts)
    #print('computed ave', np.mean(np.array(fcount_rollouts), axis=0))
    return ave_fcounts, fcount_rollouts, [steps]

    #return([np.max(learning_returns), np.min(learning_returns), np.mean(learning_returns)])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=1234, help="random seed for experiments")
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--pretrained_network', help='path to directory of pretrained network weights to form \phi(s) using all but last layer')
    parser.add_argument('--encoding_dims', type=int, help='number of dims to encode to')
    parser.add_argument('--fcount_filepath', help='path and file name to save fcount file')
    parser.add_argument('--demo_file', help="path to policy to evaluate")
    parser.add_argument('--max_length', type=int, default=50000, help="how long to run before truncating policy")

    args = parser.parse_args()
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    network_file_loc = args.pretrained_network
    print("Using network at", network_file_loc, "for features.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_net = EmbeddingNet(args.encoding_dims)
    state_dict = torch.load(network_file_loc, map_location=device)
    print(state_dict.keys())
    feature_net.load_state_dict(torch.load(network_file_loc, map_location=device))
    feature_net.to(device)

    #load the human demo
    trajectory = np.load(args.demo_file)

    print("evaluating", args.demo_file)
    ave_feature_counts, fcounts, num_steps = get_demo_feature_counts(args.env_name, trajectory, feature_net, args.max_length)
    returns = [-1]
    print("returns", returns)
    print("feature counts", ave_feature_counts)
    writer = open(args.fcount_filepath, 'w')
    utils.write_line(ave_feature_counts, writer)
    for fc in fcounts:
        utils.write_line(fc, writer)
    utils.write_line(returns, writer)
    utils.write_line(num_steps, writer, newline=False)
    writer.close()
