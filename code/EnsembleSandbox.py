import argparse
# coding: utf-8

# Use ensembles rather than Bayesian REX

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

def generate_ensemble_distribution_checkpoint(env, env_name, agent, checkpoint_model_dir, ensemble, num_rollouts, device):

    ensemble_returns = []
    true_returns = []
    # for checkpoint in checkpoints:

    model_path = checkpoint_model_dir
    #if env_name == "seaquest":
    #    model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

    agent.load(model_path)
    episode_count = num_rollouts
    for i in range(episode_count):
        ensemble_rets = np.zeros(len(ensemble))
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
            #ob_processed = ob_processed #get rid of first dimension ob.shape = (1,84,84,4)
            ob_processed = torch.from_numpy(ob_processed).float().to(device)
            for idx,net in enumerate(ensemble):
                ensemble_rets[idx] += net.cum_return(ob_processed)[0].item()

            del ob_processed
            steps += 1
            # print(steps)
            acc_reward += r[0]
            if done:
                print("checkpoint: {}, episode: {}, steps: {}, return: {}".format(model_path, i, steps,acc_reward))
                break
        true_returns.append(acc_reward)
        ensemble_returns.extend(ensemble_rets)


    return ensemble_returns, true_returns

def generate_ensemble_distribution(env, env_name, agent, model_dir, checkpoint, ensemble, num_rollouts, device):
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

    ensemble_returns = []
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
        traj_i = torch.from_numpy(traj_i).float().to(device)
        for ensemble_net in ensemble:
            cum_ret = ensemble_net.cum_return(traj_i)[0].item()
            ensemble_returns.append(cum_ret)
            #print("sample", i, "return = ", cum_ret)
        #print("traj length", len(traj))
        true_returns.append(acc_reward)


    return ensemble_returns, true_returns

#just run through one rollout since it will look the same and run extra dropouts?
def generate_ensemble_distribution_noop(env, env_name, agent, ensemble, device):

    ensemble_returns = np.zeros(len(ensemble))
    true_returns = []
    # for checkpoint in checkpoints:

    episode_count = 1
    for i in range(episode_count):
        done = False
        traj = []
        r = 0

        ob = env.reset()
        steps = 0
        acc_reward = 0
        while True and steps < 20000:
            action = 0 #no-op action
            ob, r, done, _ = env.step(action)
            ob_processed = preprocess(ob, env_name)
            #ob_processed = ob_processed #get rid of first dimension ob.shape = (1,84,84,4)
            ob_processed = torch.from_numpy(ob_processed).float().to(device)
            for d,net in enumerate(ensemble):
                ensemble_returns[d] += net.cum_return(ob_processed)[0].item()

            steps += 1
            if steps % 1000 == 0:
                print(steps)
            acc_reward += r[0]
            if done:
                print("no-op, episode: {}, steps: {}, return: {}".format(i, steps,acc_reward))
                break
        true_returns.append(acc_reward)


    return ensemble_returns, true_returns




class TrexNet(nn.Module):
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
        x = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
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
    parser.add_argument('--ensemble_models_path', default='', help="name and location for ensemble models")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    #parser.add_argument('--seed2', default=0, help="random seed for experiments after generating demos")
    parser.add_argument('--models_dir', default = None, help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    parser.add_argument('--num_traj_rollouts', default = 100, type=int, help="number of trajectories to rollout for policy eval")
    parser.add_argument('--no_op', action="store_true")
    parser.add_argument('--checkpoint_path', default = None, help='full path to the checkpoint to run Bayesian dropout on')

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

    #download the ensemble
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    import os

    ensemble_network_files = []
    with os.scandir(args.ensemble_models_path) as entries:
        for entry in entries:
            print(entry.name)
            if not '.txt' in entry.name:
                ensemble_network_files.append(os.path.join(args.ensemble_models_path, entry.name, 'reward.params'))

    print(ensemble_network_files)

    ensemble = []
    for ensemble_file in ensemble_network_files:
        ensemble_net = TrexNet()
        ensemble_net.load_state_dict(torch.load(ensemble_file, map_location=device))
        ensemble_net.to(device)
        ensemble.append(ensemble_net)


    write_directory = '../ensemble_uncertainty/'
    if args.no_op:
        pred_returns, true_returns = generate_ensemble_distribution_noop(env, env_name, agent, ensemble, device)

        f = open(write_directory + env_name + "_no_op_pred.txt", 'w')
        f2 = open(write_directory + env_name + '_no_op_true.txt', 'w')

        print(pred_returns)
        print(true_returns)
        #write data
        for p in pred_returns:
            f.write("{}\n".format(p))
        f.close()

        for t in true_returns:
            f2.write("{}\n".format(t))
        f2.close()

    elif not args.models_dir and args.checkpoint_path:

        #try and rollout with multiple passes
        pred_returns, true_returns = generate_ensemble_distribution_checkpoint(env, env_name, agent, args.checkpoint_path, ensemble, args.num_traj_rollouts, device)

        checkpoint_path_dashes = args.checkpoint_path.replace("/", "_")
        #checkpoint_path_dashes = checkpoint_path_dashes.replace(".", "_")
        f = open(write_directory + env_name + "_" + checkpoint_path_dashes + "_pred.txt", 'w')
        f2 = open(write_directory + env_name + "_" + checkpoint_path_dashes + '_true.txt', 'w')

        print(pred_returns)
        print(true_returns)
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
            pred_returns, true_returns = generate_ensemble_distribution(env, env_name, agent, args.models_dir, checkpoint, ensemble, args.num_traj_rollouts, device)

            f = open(write_directory + env_name + "_" + checkpoint + "pred.txt", 'w')
            f2 = open(write_directory + env_name + "_" + checkpoint + 'true.txt', 'w')

            # print(pred_returns)
            # print(true_returns)
            #write data
            for p in pred_returns:
                f.write("{}\n".format(p))
            f.close()

            for t in true_returns:
                f2.write("{}\n".format(t))
            f2.close()
