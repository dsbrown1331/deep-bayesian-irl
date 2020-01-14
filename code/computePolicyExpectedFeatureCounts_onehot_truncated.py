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



### Code to run policy evaluation via MC sampling from pretrained rewards using the T-rex architecture but trained on ground truth rewards for regresssion


def get_policy_feature_counts(env_name, checkpointpath, num_rollouts, max_length = 3000):
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

    print(checkpointpath)

    agent.load(checkpointpath)
    episode_count = num_rollouts

    if args.no_term:
        f_counts = np.zeros(3)  #neg, zero, pos clipped rewards
    else:
        f_counts = np.zeros(4)

    for i in range(episode_count):
        done = False
        traj = []
        r = 0

        ob = env.reset()
        #traj.append(ob)
        #print(ob.shape)
        steps = 0
        acc_reward = 0
        while steps < max_length:
            if not done:
                action = agent.act(ob, r, done)
                #print(action)
                ob, r, done, _ = env.step(action)
                ob_processed = preprocess(ob, env_name)
                #print(ob_processed.shape)
                if np.sign(r[0]) == -1:
                    if args.no_term:
                        phi_s = np.array([1.0, 0.0, 0.0])
                    else:
                        phi_s = np.array([1.0, 0.0, 0.0, 0.0])
                elif np.sign(r[0]) == 0:
                    if args.no_term:
                        phi_s = np.array([0.0, 1.0, 0.0])
                    else:
                        phi_s = np.array([0.0, 1.0, 0.0, 0.0])
                elif np.sign(r[0]) == 1:
                    if args.no_term:
                        phi_s = np.array([0.0, 0.0, 1.0])
                    else:
                        phi_s = np.array([0.0, 0.0, 1.0, 0.0])
                else:
                    print("error not a valid clipped reward")
                    sys.exit()
                #print(phi_s.shape)
                f_counts += phi_s
                steps += 1
                #print(steps)
                acc_reward += r[0]
                if done:
                    print("steps: {}, return: {}".format(steps,acc_reward))
            else:
                #add in appropriate padding and then break
                print("adding padding", max_length - steps)
                if args.no_term:
                    phi_s = (max_length - steps) * np.array([0.0, 1.0, 0.0])
                else:
                    phi_s = (max_length - steps) * np.array([0.0, 0.0, 0.0, 1.0])
                f_counts += phi_s
                print("f_counts", f_counts)

                break
        print("steps: {}, return: {}".format(steps,acc_reward))

        learning_returns.append(acc_reward)



    env.close()
    #tf.reset_default_graph()
    del agent
    del env

    ave_fcounts = f_counts/episode_count

    return learning_returns, ave_fcounts

    #return([np.max(learning_returns), np.min(learning_returns), np.mean(learning_returns)])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=1234, help="random seed for experiments")
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--output_id', help="either map or mean or number of checkpoint")
    #parser.add_argument('--checkpointpath', default='', help='path to checkpoint to run eval on')
    #parser.add_argument('--pretrained_network', help='path to pretrained network weights to form \phi(s) using all but last layer')
    parser.add_argument('--num_rollouts', type=int, help='number of rollouts to compute feature counts')
    parser.add_argument('--homedir', action='store_true', help="select this if running from my home dir on laptop")
    #parser.add_argument('--output_id', default='', help='unique id for output file name')
    parser.add_argument('--no_term', action='store_true', help = "don't use an extra terminal feature")


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



    if output_id == 'mean' or output_id == 'map':
        checkpointpath = '/scratch/cluster/dsbrown/tflogs/mcmc/' + env_name + '_linear_' + output_id + '_0/checkpoints/43000'
    elif output_id == "noop": #doesn't matter
        checkpointpath = '/scratch/cluster/dsbrown/models/' + env_name + '_25/00001'
    else:
        if args.homedir:
            checkpointpath = "/home/dsbrown/Code/learning-rewards-of-learners/learner/models/" + env_name + "_25/" + output_id
        else:
            checkpointpath = '/scratch/cluster/dsbrown/models/' + env_name + '_25/' + output_id
    print("*"*10)
    print(env_name)
    print("*"*10)
    returns, ave_feature_counts = get_policy_feature_counts(env_name, checkpointpath, args.num_rollouts)
    print("returns", returns)
    print("feature counts", ave_feature_counts)
    writer = open("../policies/" + env_name + "_" + output_id + "_fcounts_onehot_truncated_terminal" + str(args.no_term) + ".txt", 'w')
    utils.write_line(ave_feature_counts, writer)
    utils.write_line(returns, writer, newline=False)
    writer.close()
