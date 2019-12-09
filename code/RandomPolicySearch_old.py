#make sure it uses the custom baselines package
import sys
sys.path.insert(0,'./baselines/')

import argparse
# coding: utf-8

# Take as input a compressed pretrained network or run T_REX before hand
# Run randomized search over last two layers of control network that outputs action probabilities.


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
from nnet import Net
from baselines.common.trex_utils import preprocess


class PolicyNet(nn.Module):
    def __init__(self, pretrained_net, env):
        super().__init__()

        self.conv1 = pretrained_net.conv1
        self.conv2 = pretrained_net.conv2
        self.conv3 = pretrained_net.conv3
        self.conv4 = pretrained_net.conv4
        self.fc1 = nn.Linear(784, env.action_space.n)#pretrained_net.fc1
        #self.fc2 = nn.Linear(256, env.action_space.n)#(pretrained_net.fc1.out_features, env.action_space.n)
        #self.fc3 = nn.Linear(32, env.action_space.n)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    def select_action(self, obs, softmax):
        with torch.no_grad():
            '''calculate cumulative return of trajectory'''
            x = obs
            x = x.permute(0,3,1,2) #get into NCHW format
            #compute forward pass of reward network
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = x.view(-1, 784)
            #x = x.view(-1, 1936)
            #x = F.leaky_relu(self.fc1(x))
            #x = F.leaky_relu(self.fc2(x))
            logits = self.fc1(x).squeeze()
            #print(logits)
            probs = F.softmax(logits, dim=0)
            #print("probs",probs)
            if softmax:
                #choose based on probs
                action = torch.multinomial(probs, 1).item()
            else:
                #pick argmax
                _, action = probs.max(0)
                action = action.item()

            return action



def generate_fitness(env, env_name, policy, reward_fn, num_episodes, seed, render=False, softmax=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.unwrapped.envs[0].seed(seed)

    learning_returns = []
    true_returns = []
    for i in range(num_episodes):
        done = False
        traj = []
        gt_rewards = []
        r = 0

        ob = env.reset()
        ob_processed = preprocess(ob, env_name)
        #print(ob_processed.shape)
        ob_cuda = torch.from_numpy(np.array(ob_processed)).float().to(device)
        #print(ob_cuda.size())

        steps = 0
        acc_reward = 0
        true_reward = 0
        while True:

            action = policy.select_action(ob_cuda, softmax=softmax)
            #print(action)
            ob, r, done, _ = env.step(action)
            if render:
                env.render()
            ob_processed = preprocess(ob, env_name)
            #print(ob_processed.shape)
            ob_cuda = torch.from_numpy(np.array(ob_processed)).float().to(device)
            #print(ob_cuda.size())
            #ob_processed = ob_processed[0] #get rid of first dimension ob.shape = (1,84,84,4)

            steps += 1
            #print(reward_fn.predict_reward(ob_cuda).item())
            acc_reward += reward_fn.predict_reward(ob_cuda).item()
            true_reward += r
            if done or steps > 1000: #TODO: remove this if I can since it will hurt performance
                if render:
                    print("rollout: {}, steps: {}, pred return: {}, actual return {}".format(i, steps,acc_reward, true_reward))
                break
        learning_returns.append(acc_reward)
        true_returns.append(true_reward)


    return np.mean(learning_returns), np.mean(true_returns)




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




def finite_diff_search(policy_net, reward_fn, num_steps, stdev, env, env_name, num_evals, seed, step_size):
    '''hill climbing random search'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cur_policy = copy.deepcopy(policy_net)

    for i in range(num_steps):
        #need new random seed for each finite diff eval
        torch.manual_seed(seed + i * 13)
        np.random.seed(seed + i * 13)

        print()
        print("step", i)

        #create empty step direction placeholders
        step_dirs = []
        with torch.no_grad():
            for param in cur_policy.fc1.parameters():
                step_dirs.append(torch.zeros(param.size()).to(device))


        for j in range(num_evals): #number of random directions to explore
            policy_step_forward = copy.deepcopy(cur_policy)
            polciy_step_backward = copy.deepcopy(cur_policy)
            #generate noise direction
            noise = []
            with torch.no_grad():
                for param in cur_policy.fc1.parameters():
                    noise.append(torch.randn(param.size()).to(device) * stdev)

            #add random noise to weights of last two layers
            with torch.no_grad():
                for param in policy_step_forward.fc1.parameters():
                    param.add_(torch.randn(param.size()).to(device) * stdev)
                #for param in policy_proposal.fc2.parameters():
                #    print(param)
                #for param in policy_net.fc2.parameters():
                #    param.add_(torch.randn(param.size()).to(device) * stdev)

                current_fitness_forward, gt_fitness_forward = generate_fitness(env, env_name, policy_step_forward, reward_fn, 1, seed, render=False, softmax=True)

            #add random noise to weights of last two layers
            with torch.no_grad():
                for param in policy_step_backward.fc1.parameters():
                    param.add_(torch.randn(param.size()).to(device) * stdev)
                #for param in policy_proposal.fc2.parameters():
                #    print(param)
                #for param in policy_net.fc2.parameters():
                #    param.add_(torch.randn(param.size()).to(device) * stdev)

                current_fitness_backward, gt_fitness_backward = generate_fitness(env, env_name, policy_step_backward, reward_fn, 1, seed, render=False, softmax=True)

            #calc finite diff direction for gradient
            with torch.no_grad():
                for indx, layer in enumerate(step_dirs):
                    layer.add_(gt_fitness_forward - gt_fitness_backward) * noise[indx]

        #average and take step along approximate gradient
        with torch.no_grad():
            for layer in step_dirs:
                layer.divide_(num_evals)

        with torch.no_grad():
            for param in cur_policy.fc1.parameters():
                



        print("true fitness", gt_fitness)
        if gt_fitness > best_fitness:
            best_fitness = gt_fitness
            best_policy = copy.deepcopy(policy_proposal)
            print("updating best to ", best_fitness)
        else:
            print("rejecting")
    return best_policy, best_fitness




if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--num_rand_steps', default=2000, type = int, help="number of proposals to generate for MCMC")
    parser.add_argument('--rand_step_size', default = 0.1, type=float, help="proposal step is gaussian with zero mean and mcmc_step_size stdev")
    parser.add_argument('--pretrained_network', help='path to pretrained network weights to form \phi(s) using all but last layer')
    parser.add_argument('--num_rollouts', default=25, type=int, help="number of times to evaluate fitness")
    parser.add_argument('--eval', action='store_true')

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

    # Now we download a pretrained network to form \phi(s) the state features where the reward is now w^T \phi(s)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net()
    reward_net.load_state_dict(torch.load(args.pretrained_network))
    reward_net.to(device)


    #create policy net with pretrained layers from T-REX at the beginning
    policy_net = PolicyNet(reward_net, env)
    policy_net.to(device)

    #run random search over weights
    #best_reward = random_search(reward_net, demonstrations, 40, stdev = 0.01)
    if not args.eval:
        best_policy, best_perf = random_search(policy_net, reward_net, args.num_rand_steps, args.rand_step_size, env, env_name, args.num_rollouts, seed)
        pred, true = generate_fitness(env, env_name, best_policy, reward_net, 1, seed, render=False, softmax=True)
        print("pred", pred, "true", true)
        torch.save(best_policy.state_dict(), "breakout_random_hillclimbing_" + str(args.rand_step_size) +".params")
    else:
        #visualize policy learned
        policy_net.load_state_dict(torch.load("breakout_random_hillclimbing_" + str(args.rand_step_size) +".params"))
        policy_net.to(device)
        pred, true = generate_fitness(env, env_name, policy_net, reward_net, args.num_rollouts, seed, render=True, softmax=True)
