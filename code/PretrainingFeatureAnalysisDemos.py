#make sure it uses the custom baselines package
import sys
sys.path.insert(0,'./baselines/')
import warnings
warnings.filterwarnings("ignore")
import argparse
# coding: utf-8

# Take as input a compressed pretrained network and then run all the demos through the pretrained in_features
#and see what the feature counts look like as demonstrations get better


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
import utils
from nnet import Net



def generate_novice_demos(env, env_name, agent, model_dir):
    checkpoint_min = 50
    checkpoint_max = 600
    checkpoint_step = 50
    checkpoints = []
    if env_name == "enduro":
        checkpoint_min = 3100
        checkpoint_max = 3650
    elif env_name == "seaquest":
        checkpoint_min = 10
        checkpoint_max = 65
        checkpoint_step = 5
    for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints.append('0000' + str(i))
        elif i < 100:
            checkpoints.append('000' + str(i))
        elif i < 1000:
            checkpoints.append('00' + str(i))
        elif i < 10000:
            checkpoints.append('0' + str(i))
    print(checkpoints)



    demonstrations = []
    learning_returns = []
    learning_rewards = []
    for checkpoint in checkpoints:

        model_path = model_dir + env_name + "_25/" + checkpoint
        if env_name == "seaquest":
            model_path = model_dir + env_name + "_5/" + checkpoint

        agent.load(model_path)
        episode_count = 1
        for i in range(episode_count):
            done = False
            traj = []
            gt_rewards = []
            r = 0

            ob = env.reset()
            steps = 0
            acc_reward = 0
            while True:
                action = agent.act(ob, r, done)
                ob, r, done, _ = env.step(action)
                ob_processed = preprocess(ob, env_name)
                #ob_processed = ob_processed[0] #get rid of first dimension ob.shape = (1,84,84,4)
                traj.append(ob_processed)

                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]
                if done:
                    print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                    break
            print("traj length", len(traj))
            print("demo length", len(demonstrations))
            demonstrations.append(traj)
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)

    return demonstrations, learning_returns, learning_rewards

def generate_feature_counts(demos, reward_net):
    feature_cnts = torch.zeros(len(demos), reward_net.fc2.in_features) #no bias
    for i in range(len(demos)):
        traj = np.array(demos[i])
        traj = torch.from_numpy(traj).float().to(device)
        #print(len(trajectory))
        feature_cnts[i,:] = reward_net.state_features(traj).squeeze().float().to(device)
    return feature_cnts.to(device)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--models_dir', default = ".", help="path to directory that contains checkpoint models for demos are stored")
    parser.add_argument('--pretrained_network', help='path to pretrained network weights to form \phi(s) using all but last layer')
    parser.add_argument('--encoding_dims', help="size of latent space", type=int)
    parser.add_argument('--trex', action='store_true', help='use icml trex pretrained feature network with bias')

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

    demonstrations, learning_returns, learning_rewards = generate_novice_demos(env, env_name, agent, args.models_dir)

    #sort the demonstrations according to ground truth reward to simulate ranked demos

    print([a[0] for a in zip(learning_returns, demonstrations)])
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)
    print("lengths")
    print([len(d) for d in demonstrations])


    # Now we download a pretrained network to form \phi(s) the state features where the reward is now w^T \phi(s)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.trex:
        print("using TREX network from ICML")
        reward_net = Net()
    else:
        reward_net = EmbeddingNet(args.encoding_dims)
    reward_net.load_state_dict(torch.load(args.pretrained_network, map_location=device))
    #reinitialize last layer
    num_features = reward_net.fc2.in_features

    print("reward is linear combination of ", num_features, "features")
    reward_net.to(device)
    #freeze all weights so there are no gradients (we'll manually update the last layer via proposals so no grads required)
    for param in reward_net.parameters():
        param.requires_grad = False

    #get num_demos by num_features + 1 (bias) numpy array with (un-discounted) feature counts from pretrained network
    directories = args.pretrained_network.split("/") #split on directories to get the last past
    filename = directories[-1] #last element should be the name of the pretrained network
    fname = filename.split(".")[0] #get first part before the .params_...
    demo_cnts = generate_feature_counts(demonstrations, reward_net) #compute the fcounts
    demo_cnts = demo_cnts.cpu().numpy() #get off gpu and into numpy array
    fcount_out_filename = "../feature_analysis/" + fname + "_fcount_demos.txt"
    print("writing to ", fcount_out_filename)

    writer = open(fcount_out_filename, 'w')
    for i,fc in enumerate(demo_cnts):
        if i < len(demo_cnts) - 1:
            utils.write_line(fc, writer)
        else:
            utils.write_line(fc, writer, newline=False)
    writer.close()
