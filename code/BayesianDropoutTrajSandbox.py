import argparse
# coding: utf-8

# Take length 50 snippets and record the cumulative return for each one. Then determine ground truth labels based on this.

# In[1]:


import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *
from baselines.common.trex_utils import preprocess

def generate_dropout_distribution_checkpoint(env, env_name, agent, checkpoint_model_dir, dropout_net, num_rollouts, num_dropout_samples, device, time_limit=100000):

    dropout_returns = []
    true_returns = []
    # for checkpoint in checkpoints:

    model_path = checkpoint_model_dir
    #if env_name == "seaquest":
    #    model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint


    agent.load(model_path)
    episode_count = num_rollouts
    for i in range(episode_count):
        dropout_rets = np.zeros(num_dropout_samples)

        dropout_masks = []

        done = False
        traj = []
        r = 0

        ob = env.reset()
        steps = 0
        acc_reward = 0
        while True and steps < time_limit:
            action = agent.act(ob, r, done)
            ob, r, done, _ = env.step(action)
            ob_processed = preprocess(ob, env_name)
            if steps == 0:
                #generate masks one for each dropout and keep them fixed for trajectories
                for d in range(num_dropout_samples):
                    dropout_masks.append( dropout_net.cum_return(ob_processed)[2] )

            #ob_processed = ob_processed #get rid of first dimension ob.shape = (1,84,84,4)
            ob_processed = torch.from_numpy(ob_processed).float().to(device)
            for d in range(num_dropout_samples):
                dropout_rets[d] += dropout_net.cum_return(ob_processed, mask=dropout_masks[d])[0].item()

            del ob_processed
            steps += 1
            #print(steps)
            acc_reward += r[0]
            if done:
                print("checkpoint: {}, episode: {}, steps: {}, return: {}".format(model_path, i, steps,acc_reward))
                break
        if steps >= time_limit:
            print("checkpoint: {}, episode: {}, steps: {}, return: {}".format(model_path, i, steps,acc_reward))

        true_returns.append(acc_reward)
        dropout_returns.extend(dropout_rets)


    return dropout_returns, true_returns

def generate_dropout_distribution_framestack(env, env_name, framestack_path, dropout_net, num_dropout_samples, device, time_limit=100000):
    #uses a prerecorded framestack to do return uncertainty analysis on
    dropout_rets = np.zeros(num_dropout_samples)
    true_returns = [-1] #TODO: I don't have a way to get true returns yet. Need to grab these from Prabhat's code. Should be able to get from rewards saved

    #load the framestack
    trajectory = np.load(framestack_path)

    #generate masks one for each dropout and keep them fixed for trajectories
    dropout_masks = []

    for i in range(min(time_limit, len(trajectory))):
        ob = trajectory[i]
        ob_processed = preprocess(ob, env_name)
        ob_processed = torch.from_numpy(ob_processed).float().to(device)

        if i == 0:
            for d in range(num_dropout_samples):
                dropout_masks.append( dropout_net.cum_return(ob_processed)[2] )

        for d in range(num_dropout_samples):
            dropout_rets[d] += dropout_net.cum_return(ob_processed, mask=dropout_mask[d])[0].item()

        del ob_processed

    #true_returns.append(acc_reward) #TODO



    return dropout_rets, true_returns




def generate_dropout_distribution(env, env_name, agent, model_dir, checkpoint, dropout_net, num_rollouts, num_dropout_samples, device):
    # checkpoints = []
    # checkpts = [500]
    # for i in checkpts:
    #     if i < 10:
    #         checkpoints.append('0000' + str(i))
    #     elif i < 100:
    #         checkpoints.append('000' + str(i))
    #     elif i < 1000:
    #         checkpoints.append('00' + str(i))
    #     elif i < 10000:
    #         checkpoints.append('0' + str(i))
    # print(checkpoints)

    dropout_returns = []
    true_returns = []
    # for checkpoint in checkpoints:

    model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
    #if env_name == "seaquest":
    #    model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

    agent.load(model_path)
    episode_count = num_rollouts
    for i in range(episode_count):
        done = False
        traj = []
        r = 0


        ob = env.reset()
        steps = 0
        acc_reward = 0
        while True:
            action = agent.act(ob, r, done)
            ob, r, done, _ = env.step(action)
            ob_processed = preprocess(ob, env_name)
            ob_processed = ob_processed[0] #get rid of first dimension ob.shape = (1,84,84,4)
            traj.append(ob_processed)

            steps += 1
            acc_reward += r[0]
            if done:
                print("checkpoint: {}, episode: {}, steps: {}, return: {}".format(checkpoint, i, steps,acc_reward))
                break
        #now run the traj through the network
        #convert to pytorch tensor
        traj_i = np.array(traj)
        traj_i = torch.from_numpy(traj_i[::2]).float().to(device)  #skip out on every other framestack to speed up but not lose information
        for i in range(num_dropout_samples):
            cum_ret = dropout_net.cum_return(traj_i)[0].item()
            dropout_returns.append(cum_ret)
            #print("sample", i, "return = ", cum_ret)
        #print("traj length", len(traj))
        true_returns.append(acc_reward)


    return dropout_returns, true_returns

#just run through one rollout since it will look the same and run extra dropouts?
def generate_dropout_distribution_noop(env, env_name, agent, dropout_net, num_dropout_samples, device):

    dropout_returns = np.zeros(num_dropout_samples)
    true_returns = []
    # for checkpoint in checkpoints:

    episode_count = 1
    for i in range(episode_count):
        done = False
        traj = []
        r = 0

        #generate masks one for each dropout and keep them fixed for trajectories
        dropout_masks = []

        ob = env.reset()
        steps = 0
        acc_reward = 0
        while True and steps < 20000:
            action = 0 #no-op action
            ob, r, done, _ = env.step(action)
            ob_processed = preprocess(ob, env_name)
            #ob_processed = ob_processed #get rid of first dimension ob.shape = (1,84,84,4)
            ob_processed = torch.from_numpy(ob_processed).float().to(device)

            if steps == 0:
                for d in range(num_dropout_samples):
                    dropout_masks.append( dropout_net.cum_return(ob_processed)[2] )

            for d in range(num_dropout_samples):
                dropout_returns[d] += dropout_net.cum_return(ob_processed, mask=dropout_masks[d])[0].item()

            steps += 1
            if steps % 1000 == 0:
                print(steps)
            acc_reward += r[0]
            if done:
                print("noop:, episode: {}, steps: {}, return: {}".format(i, steps,acc_reward))
                break
        true_returns.append(acc_reward)


    return dropout_returns, true_returns




class DropoutNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 1)



    def cum_return(self, traj, mask=None):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        x = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 784)

        #if mask is given then use it, otherwise generate a new mask for all states in the trajectory
        if mask is None:
            my_mask = (torch.rand(64) < 0.5).float().to(device) / 0.5
        else:
            my_mask = mask
        #apply the mask to the entire trajectory.
        x = F.leaky_relu(self.fc1(x)) * my_mask
        r = self.fc2(x)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards, my_mask



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, mask = self.cum_return(traj_i)
        cum_r_j, abs_r_j, _ = self.cum_return(traj_j, mask)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j





# Train the network
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()

    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            loss = loss_criterion(outputs, labels.long()) + l1_reg * abs_rewards
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 100 == 99:
                #print(i)
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                print(abs_rewards)
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_net.state_dict(), checkpoint_dir)
    print("finished training")





def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)






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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--reward_model_path', default='', help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    #parser.add_argument('--seed2', default=0, help="random seed for experiments after generating demos")
    parser.add_argument('--models_dir', default = None, help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    parser.add_argument('--num_traj_rollouts', default = 100, type=int, help="number of trajectories to rollout for policy eval")
    parser.add_argument('--num_dropout_samples', default = 100, type = int, help = "number of mcmc_dropouts to perform on a trajectory")
    parser.add_argument('--no_op', action="store_true", help="run no-op policy eval")
    parser.add_argument('--checkpoint_path', default = None, help='full path to the checkpoint or human demo framestack to run Bayesian dropout on')
    parser.add_argument('--time_limit', type=int, default = 100000, help= 'use fixed horizon evaluation, only works for checkpoint_path != None')
    parser.add_argument('--human_demo', action="store_true", help='run bayesian dropout on human demo framestack')

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dropout_net = DropoutNet()
    dropout_net.load_state_dict(torch.load(args.reward_model_path, map_location=device))
    dropout_net.to(device)

    write_directory = '../bayesian_dropout_traj/'

    if args.no_op:
        pred_returns, true_returns = generate_dropout_distribution_noop(env, env_name, agent, dropout_net, args.num_dropout_samples, device)

        f = open(write_directory + env_name + "_no_op_pred.txt", 'w')
        f2 = open(write_directory + env_name + '_no_op_true.txt', 'w')

        #write data
        for p in pred_returns:
            f.write("{}\n".format(p))
        f.close()

        for t in true_returns:
            f2.write("{}\n".format(t))
        f2.close()

    elif not args.models_dir and args.checkpoint_path and not args.human_demo:
        print(args.time_limit)
        #try and rollout with multiple passes
        pred_returns, true_returns = generate_dropout_distribution_checkpoint(env, env_name, agent, args.checkpoint_path, dropout_net, args.num_traj_rollouts, args.num_dropout_samples, device, args.time_limit)

        checkpoint_path_dashes = args.checkpoint_path.replace("/", "_")
        #checkpoint_path_dashes = checkpoint_path_dashes.replace(".", "_")
        f = open(write_directory + env_name + "_" + checkpoint_path_dashes + "_pred.txt", 'w')
        f2 = open(write_directory + env_name + "_" + checkpoint_path_dashes + '_true.txt', 'w')

        #write data
        for p in pred_returns:
            f.write("{}\n".format(p))
        f.close()

        for t in true_returns:
            f2.write("{}\n".format(t))
        f2.close()

    elif args.human_demo:
        print("running Bayesian dropout on recorded human demo")

        pred_returns, true_returns = generate_dropout_distribution_framestack(env, env_name, args.checkpoint_path, dropout_net, args.num_dropout_samples, device, args.time_limit)

        checkpoint_path_dashes = args.checkpoint_path.replace("/", "_")
        #checkpoint_path_dashes = checkpoint_path_dashes.replace(".", "_")
        if args.time_limit != 100000:
            #write the time_limit to the end of the file to differentiate it
            f = open(write_directory + env_name + "_" + checkpoint_path_dashes + "_" + str(args.time_limit) + "_pred.txt", 'w')
            f2 = open(write_directory + env_name + "_" + checkpoint_path_dashes + "_" + str(args.time_limit) +  '_true.txt', 'w')
        else:
            f = open(write_directory + env_name + "_" + checkpoint_path_dashes + "_pred.txt", 'w')
            f2 = open(write_directory + env_name + "_" + checkpoint_path_dashes + '_true.txt', 'w')

        #write data
        for p in pred_returns:
            f.write("{}\n".format(p))
        f.close()

        for t in true_returns:
            f2.write("{}\n".format(t))
        f2.close()



    else:

        if env_name == "enduro":
            checkpoints = ['03125', '03425', '03900', '04875']
        else:
            checkpoints = ['00025', '00325', '00800', '01450']

        for checkpoint in checkpoints:

            #try and rollout with multiple passes
            pred_returns, true_returns = generate_dropout_distribution(env, env_name, agent, args.models_dir, checkpoint, dropout_net, args.num_traj_rollouts, args.num_dropout_samples, device)

            f = open(write_directory + env_name + "_" + checkpoint + "pred.txt", 'w')
            f2 = open(write_directory + env_name + "_" + checkpoint + 'true.txt', 'w')

            #write data
            for p in pred_returns:
                f.write("{}\n".format(p))
            f.close()

            for t in true_returns:
                f2.write("{}\n".format(t))
            f2.close()
