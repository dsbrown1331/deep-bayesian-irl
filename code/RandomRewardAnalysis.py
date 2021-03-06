import argparse
# coding: utf-8
#This experiment is to see if random rewards are always decent for Breakout, or if
#they sometimes give all negative rewards and are not good. I'm hoping and thinking that
#if I generate a couple random rewards that there should be both positive and negative outputs


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
from baselines.common.trex_utils import preprocess


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



def create_training_data(demonstrations):
    '''create all pairwise rankings given list of sorted demonstrations'''
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)
    for i in range(num_demos):
        for j in range(i+1,num_demos):
            #print(i,j)
            traj_i = demonstrations[i]
            traj_j = demonstrations[j]
            label = 1
            training_obs.append((traj_i, traj_j))
            training_labels.append(label)

    return training_obs, training_labels






class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)



    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        for x in traj:
            x = x.permute(0,3,1,2) #get into NCHW format
            #compute forward pass of reward network
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = x.view(-1, 784)
            #x = x.view(-1, 1936)
            x = F.leaky_relu(self.fc1(x))
            #r = torch.tanh(self.fc2(x)) #clip reward?
            r = self.fc2(x)
            #r = torch.sigmoid(r) #TODO: try without this
            sum_rewards += r
        ##    y = self.scalar(torch.ones(1))
        ##    sum_rewards += y
        #print(sum_rewards)
        return sum_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        #print([self.cum_return(traj_i), self.cum_return(traj_j)])
        cum_r_i = self.cum_return(traj_i)
        cum_r_j = self.cum_return(traj_j)
        #print(abs_r_i + abs_r_j)
        return cum_r_i, cum_r_j



def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            rewards_from_obs.append(r)
    return rewards_from_obs

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))

def print_traj_returns(reward_net, demonstrations):
    #print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

def calc_pairwise_ranking_loss(reward_net, demo_pairs, preference_labels):
    '''sum over all pairwise demonstrations the softmax loss from T-REX and Christiano work'''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    #don't need any gradients
    with torch.no_grad():

        loss_criterion = nn.CrossEntropyLoss()
        cum_log_likelihood = 0.0
        for i in range(len(preference_labels)):
            traj_i, traj_j = demo_pairs[i]
            labels = np.array([preference_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #just need forward pass and loss calculation
            return_i, return_j = reward_net.forward(traj_i, traj_j)
            #print(return_i, return_j)
            #outputs = torch.from_numpy(np.array([return_i.item(), return_j.item()])).float().to(device)
            outputs = torch.cat([return_i, return_j], dim=1)

            #outputs = outputs.unsqueeze(0)
            #print(outputs)
            #print(labels)
            log_likelihood = -loss_criterion(outputs, labels)
            #if labels == 0:
            #    log_likelihood = torch.log(return_i/(return_i + return_j))
            #else:
            #    log_likelihood = torch.log(return_j/(return_i + return_j))
            #print("ll",log_likelihood)
            cum_log_likelihood += log_likelihood
    return cum_log_likelihood


def random_search(reward_net, demonstrations, num_trials, stdev = 0.1):
    '''hill climbing random search'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_likelihood = -np.inf
    best_reward = copy.deepcopy(reward_net)
    #create the pairwise rankings for loss calculations
    demo_pairs, preference_labels = create_training_data(demonstrations)
    for i in range(num_trials):
        print()
        print("trial", i)
        reward_net_proposal = copy.deepcopy(best_reward)
        #add random noise to weights
        with torch.no_grad():
            for param in reward_net_proposal.parameters():
                param.add_(torch.randn(param.size()).to(device) * stdev)
        #print_traj_returns(reward_net_proposal, demonstrations)
        cum_loglik = calc_pairwise_ranking_loss(reward_net_proposal, demo_pairs, preference_labels)
        print("pair-wise ranking loglik", cum_loglik)
        if cum_loglik > best_likelihood:
            best_likelihood = cum_loglik
            best_reward = copy.deepcopy(reward_net_proposal)
            print("updating best to ", best_likelihood)
        else:
            print("rejecting")
    return best_reward


def mcmc_map_search(reward_net, demonstrations, num_steps, step_stdev = 0.1):
    '''try out a bunch of random weights and see what the resulting predicted returns look like'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    demo_pairs, preference_labels = create_training_data(demonstrations)

    starting_loglik = calc_pairwise_ranking_loss(reward_net, demo_pairs, preference_labels)

    map_loglik = starting_loglik
    map_reward = copy.deepcopy(reward_net)

    cur_reward = copy.deepcopy(reward_net)
    cur_loglik = starting_loglik

    reject_cnt = 0
    accept_cnt = 0

    for i in range(num_steps):
        print()
        print("step", i)
        #take a proposal step
        proposal_reward = copy.deepcopy(cur_reward)
        #add random noise to weights
        with torch.no_grad():
            for param in proposal_reward.parameters():
                param.add_(torch.randn(param.size()).to(device) * step_stdev)
        #debugging info
        print_traj_returns(proposal_reward, demonstrations)
        #calculate prob of proposal
        prop_loglik = calc_pairwise_ranking_loss(proposal_reward, demo_pairs, preference_labels)
        print("proposal loglik", prop_loglik.item())
        print("cur loglik", cur_loglik.item())
        if prop_loglik > cur_loglik:
            #accept always
            print("accept")
            accept_cnt += 1
            cur_reward = copy.deepcopy(proposal_reward)
            cur_loglik = prop_loglik

            #check if this is best so far
            if prop_loglik > map_loglik:
                map_loglik = prop_loglik
                map_reward = copy.deepcopy(proposal_reward)
                print("updating map to ", prop_loglik)
        else:
            #accept with prob exp(prop_loglik - cur_loglik)
            if np.random.rand() < torch.exp(prop_loglik - cur_loglik).item():
                print("probabilistic accept")
                accept_cnt += 1
                cur_reward = copy.deepcopy(proposal_reward)
                cur_loglik = prop_loglik
            else:
                #reject and stick with cur_reward
                print("reject")
                reject_cnt += 1
                continue
    print("num rejects", reject_cnt)
    print("num accepts", accept_cnt)

    return map_reward

def compute_l1(last_layer):
    linear, bias = list(last_layer.parameters())
    #print(linear)
    #print(bias)
    with torch.no_grad():
        weights = torch.cat((linear.squeeze(), bias)).cpu().numpy()
    #print("output", np.sum(np.abs(weights)))
    return np.sum(np.abs(weights))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--reward_model_path', default='', help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--num_seeds', default=0, help="random seed for experiments")
    parser.add_argument('--models_dir', default = ".", help="path to directory that contains checkpoint models for demos are stored")
    parser.add_argument('--num_trajs', default = 0, type=int, help="number of downsampled full trajectories")
    parser.add_argument('--num_snippets', default = 6000, type = int, help = "number of short subtrajectories to sample")
    parser.add_argument('--pretrained_network', help='path to pretrained network weights to form \phi(s) using all but last layer')
    parser.add_argument('--mcmc_step_size', default = 0.005, type=float, help="proposal step is gaussian with zero mean and mcmc_step_size stdev")

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
    demo_seed = 0 #fixed to reduce variability


    stochastic = True


    env = make_vec_env(env_id, 'atari', 1, demo_seed,
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

    #random seed for MCMC
    num_seeds = int(args.num_seeds)
    for seed in range(num_seeds):
        print("*"*10)
        print("Seed", seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        # Now we create a reward network and optimize it using the training data.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        reward_net = Net()
        reward_net.load_state_dict(torch.load(args.pretrained_network))
        reward_net.fc2 = nn.Linear(reward_net.fc2.in_features, 1)
        reward_net.to(device)
        last_layer = reward_net.fc2
        with torch.no_grad():
            for param in last_layer.parameters():
                param.add_(torch.randn(param.size()).to(device) * args.mcmc_step_size)
        l1_norm = np.array([compute_l1(last_layer)])
        #normalize the weight vector...
        with torch.no_grad():
            for param in last_layer.parameters():
                param.div_(torch.from_numpy(l1_norm).float().to(device))

        #run random search over weights
        #best_reward = random_search(reward_net, demonstrations, 40, stdev = 0.01)
        best_reward = reward_net#mcmc_map_search(reward_net, demonstrations, args.num_mcmc_steps, step_stdev = args.mcmc_step_size)
        #save best reward network
        #torch.save(best_reward.state_dict(), args.reward_model_path)
        demo_pairs, preference_labels = create_training_data(demonstrations)
        print("best reward ll", calc_pairwise_ranking_loss(best_reward, demo_pairs, preference_labels))
        print_traj_returns(best_reward, demonstrations)


    #add random
