### This code will take in any pretrained network and compute the expected feature counts via Monte Carlo sampling according to the last
### layer of the pretrained network


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




def get_policy_feature_counts(env_name, checkpointpath, feature_net, num_rollouts, no_op=False):
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    elif env_name == "montezumarevenge":
        env_id = "MontezumaRevengeNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

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
    fcount_rollouts = [] #keep track of the feature counts for each rollout
    num_steps = []

    print("using checkpoint", checkpointpath, "if none then using no-op policy")
    if not no_op:
        agent.load(checkpointpath)
    episode_count = num_rollouts

    f_counts = np.zeros(feature_net.fc2.in_features)


    for i in range(episode_count):
        done = False
        traj = []
        fc_rollout = np.zeros(feature_net.fc2.in_features)
        r = 0

        ob = env.reset()
        #traj.append(ob)
        #print(ob.shape)
        steps = 0
        acc_reward = 0
        while True:#steps < 7000:
            if no_op:
                action = 0
            else:
                action = agent.act(ob, r, done)
            #print(action)
            ob, r, done, _ = env.step(action)
            ob_processed = preprocess(ob, env_name)
            #print(ob_processed.shape)
            phi_s = feature_net.state_feature(torch.from_numpy(ob_processed).float().to(device)).cpu().squeeze().numpy()
            #print(phi_s.shape)
            fc_rollout += phi_s
            f_counts += phi_s
            steps += 1
            #print(steps)
            acc_reward += r[0]
            if done:
                print("steps: {}, return: {}".format(steps,acc_reward))
                break
        fcount_rollouts.append(fc_rollout)
        learning_returns.append(acc_reward)
        num_steps.append(steps)



    env.close()
    #tf.reset_default_graph()
    del agent
    del env

    ave_fcounts = f_counts/episode_count
    print('ave', ave_fcounts)
    print('computed ave', np.mean(np.array(fcount_rollouts), axis=0))
    return learning_returns, ave_fcounts, fcount_rollouts, num_steps

    #return([np.max(learning_returns), np.min(learning_returns), np.mean(learning_returns)])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=1234, help="random seed for experiments")
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--output_id', help="either map or mean or number of checkpoint")
    #parser.add_argument('--checkpointpath', default='', help='path to checkpoint to run eval on')
    parser.add_argument('--pretrained_network_dir', help='path to directory of pretrained network weights to form \phi(s) using all but last layer')
    parser.add_argument('--num_rollouts', type=int, help='number of rollouts to compute feature counts')
    parser.add_argument('--postfix', default='', help='postfix of network files, all are assumed to be of form "[env_name]postfix" and are located in the pretrained_network_dir')
    parser.add_argument('--encoding_dims', type=int, help='number of dims to encode to')
    parser.add_argument('--fcount_dir', help='directory to save fcount file')
    parser.add_argument('--no_op', action='store_true', help='run no-op policy')
    parser.add_argument('--rl_eval', action='store_true', help='use rl policies on scratch to do eval, do a lot of them')

    args = parser.parse_args()
    env_name = args.env_name
    output_id = args.output_id
    #output_ids = ['00025', '00325', '00800', '01450', 'mean', 'map']
    #if env_name == 'enduro':
    #    output_ids = ['03125', '03425', '03900', '04875', 'mean', 'map']
    print("generating feature counts for",output_id)
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    network_file_loc = os.path.abspath(os.path.join(args.pretrained_network_dir, args.env_name + args.postfix))
    print("Using network at", network_file_loc, "for features.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_net = EmbeddingNet(args.encoding_dims)
    state_dict = torch.load(network_file_loc, map_location=device)
    print(state_dict.keys())
    feature_net.load_state_dict(torch.load(network_file_loc, map_location=device))
    feature_net.to(device)

    if args.no_op:
        checkpointpath = None
    elif args.rl_eval:
        checkpointpath = '/scratch/cluster/dsbrown/tflogs/rl/' + env_name + '_0/checkpoints/' + output_id
    elif output_id == 'map':
        #checkpointpath = '/scratch/cluster/dsbrown/tflogs/mcmc/' + env_name + '_linear_' + output_id + '_0/checkpoints/43000'
        checkpointpath = '/scratch/cluster/dsbrown/tflogs/mcmc/' + env_name + '_64_all/checkpoints/43000'
    elif output_id == 'mean':
        #checkpointpath = '/scratch/cluster/dsbrown/tflogs/mcmc/' + env_name + '_linear_' + output_id + '_0/checkpoints/43000'
        checkpointpath = '/scratch/cluster/dsbrown/tflogs/mcmc/' + env_name + '_64_all_mean/checkpoints/43000'
    else:
        #checkpointpath = '../../learning-rewards-of-learners/learner/models/' + env_name + '_25/' + output_id
        checkpointpath = '/scratch/cluster/dsbrown/models/' + env_name + '_25/' + output_id
    print("evaluating", checkpointpath)
    print("*"*10)
    print(env_name)
    print("*"*10)
    returns, ave_feature_counts, fcounts, num_steps = get_policy_feature_counts(env_name, checkpointpath, feature_net, args.num_rollouts, args.no_op)
    print("returns", returns)
    print("feature counts", ave_feature_counts)
    writer = open(args.fcount_dir + env_name + "_" + output_id + args.postfix + "_fcounts_auxiliary.txt", 'w')
    utils.write_line(ave_feature_counts, writer)
    for fc in fcounts:
        utils.write_line(fc, writer)
    utils.write_line(returns, writer)
    utils.write_line(num_steps, writer, newline=False)
    writer.close()
