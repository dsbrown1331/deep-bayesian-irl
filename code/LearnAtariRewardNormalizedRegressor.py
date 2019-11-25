import argparse
# Learn to regress from pixels to clipped rewards from observations
# I oversample the non-zero observations to balance the data since usually about 80-90% of the observations have zero reward
# This might be causing false positives, though.

import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *
from baselines.common.trex_utils import mask_score

from nets import TrexNet, MediumNet, BiggerNet


def generate_training_data(env, env_name, agent, model_dir, episode_count, debug):
    checkpoint_min = 1000#50
    checkpoint_max = 1400#1450
    checkpoint_step = 100#50
    checkpoints = []
    crop_top = True
    if env_name == "enduro":
        checkpoint_min = 3225
        checkpoint_max = 4825
        checkpoint_step = 400
        crop_top = False
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



    states = []
    rewards = []
    total_returns = 0.0
    for checkpoint in checkpoints:

        model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
        if env_name == "seaquest":
            model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

        agent.load(model_path)

        for i in range(episode_count):
            done = False
            r = 0

            ob = env.reset()
            #traj.append(ob)
            #print(ob.shape)
            steps = 0
            acc_reward = 0
            while True:
                action = agent.act(ob, r, done)
                ob, r, done, _ = env.step(action)
                #print(ob.shape)
                masked_ob = mask_score(ob, env_name)
                states.append(masked_ob[0])
                rewards.append(r[0])
                if debug:
                    if r[0] != 0:
                        import matplotlib.pyplot as plt
                        print(masked_ob[0].shape)
                        for cnt in range(4):
                            plt.subplot(1,4,cnt+1)
                            plt.imshow(masked_ob[0][:,:,cnt])
                            plt.axis('off')
                        plt.title("REWARD = {}".format(r[0]))
                        plt.show()
                steps += 1
                acc_reward += r[0]
                if done:
                    print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                    total_returns += acc_reward
                    break
    print(total_returns)
    return states, rewards


#generate data from a better policy
def generate_training_data_birlmap(env, env_name, agent, episode_count, debug, path, map = True):

    states = []
    rewards = []
    total_returns = 0.0
    model_path = path
    agent.load(model_path)

    for i in range(episode_count):
        done = False
        r = 0

        ob = env.reset()
        #traj.append(ob)
        #print(ob.shape)
        steps = 0
        acc_reward = 0
        while True:
            if np.random.rand() < 0.02: #random noise in case policy gets stuck
                action = env.action_space.sample()
            else:
                action = agent.act(ob, r, done)
            ob, r, done, _ = env.step(action)
            #print(ob.shape)
            masked_ob = mask_score(ob, env_name)
            states.append(masked_ob[0])
            rewards.append(r[0])
            if debug:
                if r[0] != 0:
                    import matplotlib.pyplot as plt
                    print(masked_ob[0].shape)
                    for cnt in range(4):
                        plt.subplot(1,4,cnt+1)
                        plt.imshow(masked_ob[0][:,:,cnt])
                        plt.axis('off')
                    plt.title("REWARD = {}".format(r[0]))
                    plt.show()
            steps += 1
            acc_reward += r[0]
            if done:
                print("episode {}, steps: {}, return: {}".format(i, steps,acc_reward))
                total_returns += acc_reward
                break
    print(total_returns)
    return states, rewards







'''learn classifier for whether a reward is zero or one'''
def learn_reward_classifier(reward_network, optimizer, training_inputs, training_outputs, num_iter, batch_size, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.MSELoss()
    #print(training_data[0])
    #print('tinput', type(training_inputs))
    best_error = np.inf
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        cum_loss = 0.0
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        #print('tobs', type(training_obs))
        for i in range(0,len(training_labels),batch_size):
            #print(i)
            ##normalize here rather than when we create to see if it cuts down on size
            obs = np.array(training_obs[i:i+batch_size]) / 255.0
            labels = np.array(training_labels[i:i+batch_size])

            #print(obs)
            #print(type(obs))
            #print(obs.shape)
            #print('obs size', obs.shape)
            #print('labels size', labels.shape)
            obs = torch.from_numpy(obs).float().to(device)
            labels = torch.from_numpy(labels).float().to(device)
            #print('obs size', obs.size())
            #print('labels size', labels.size())
            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs = reward_network.forward(obs)
            #outputs = outputs.unsqueeze(0)
            #print("outputs", outputs)
            #print("labels", labels)

            loss = loss_criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
        print('-'*20)
        print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
        val_accuracy_all = calc_accuracy(reward_net, validation_obs, validation_labels)
        print("Validation Error All= ", val_accuracy_all)
        if val_accuracy_all < best_error:
            print("lowest regression error so far")
            best_error = val_accuracy_all
            torch.save(reward_net.state_dict(), args.reward_model_path)
            #print('loss', item_loss)
            # cum_loss += item_loss
            # if i % 500 == 499:
            #     #print(i)
            #     print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
            #     print(abs_rewards)
                #     cum_loss = 0.0
            #     print("check pointing")
            #     torch.save(reward_net.state_dict(), checkpoint_dir)
    print("finished training")





def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    error = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            obs  = training_inputs[i]
            obs  = np.array([obs]) / 255.0
            obs = torch.from_numpy(obs).float().to(device)

            #forward to get logits
            output = reward_network.forward(obs).item()
            error += (label - output) * (label - output)
    return error / len(training_inputs)






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
    parser.add_argument('--reward_model_path', default='', help="name and location for learned model params")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--models_dir', default = ".", help="top directory where checkpoint models for demos are stored")
    parser.add_argument('--episode_count', default = 5, type=int, help = "how many rollouts per checkpoint for generating training data")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--val_prop', default=0.2, type=float, help="what proportion of training data to use as validation")
    parser.add_argument('--policy_path', default=None, type=str, help="path to use to generate better training data")

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

    print("Training reward for", env_id)
    lr = 0.0001
    weight_decay = 0.0001
    num_iter = 20 #num times through training data
    batch_size = 32
    stochastic = True

    #env id, env type, num envs, and seed
    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })


    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)

    tstates, trewards = generate_training_data(env, env_name, agent, args.models_dir, args.episode_count, args.debug)
    if args.policy_path:
        more_states, more_rewards = generate_training_data_birlmap(env, env_name, agent, args.episode_count, args.debug, args.policy_path)
        tstates = tstates + more_states
        trewards = trewards + more_rewards
    print(len(tstates))
    print(len(trewards))
    print("reward values", set(trewards))



    #check how imbalanced the data is
    zero_sr_pairs = []
    nonzero_sr_pairs = []
    for i,r in enumerate(trewards):
        if r == 0:
            zero_sr_pairs.append((tstates[i], trewards[i]))
        else:
            #nonzero_sr_pairs.append((tstates[i], trewards[i]))
            nonzero_sr_pairs.append((tstates[i], np.sign(trewards[i])))
    print(len(zero_sr_pairs))
    print(len(nonzero_sr_pairs))
    #shuffle before assinging to validation /train
    np.random.shuffle(zero_sr_pairs)
    np.random.shuffle(nonzero_sr_pairs)

    #separate into validation and train and (test?)
    num_val_zero = int(args.val_prop*len(zero_sr_pairs))
    num_val_nonzero = int(args.val_prop*len(nonzero_sr_pairs))
    print("validation_sizes", num_val_zero, num_val_nonzero)
    validation_data_zero = zero_sr_pairs[:num_val_zero]
    validation_data_nonzero = nonzero_sr_pairs[:num_val_nonzero]
    training_data_zero = zero_sr_pairs[num_val_zero:]
    training_data_nonzero = nonzero_sr_pairs[num_val_nonzero:]

    #how to balance dataset? I'm going to just duplicate for now
    training_data_nonzero = random.choices(training_data_nonzero,k=len(training_data_zero))

    #combine and shuffle to create train and validation
    training_data = training_data_zero + training_data_nonzero


    training_obs, training_labels = zip(*training_data)
    training_obs = np.array(training_obs)
    training_labels = np.array(training_labels)

    validation_obs_zero, validation_labels_zero = zip(*validation_data_zero)
    validation_obs_zero = np.array(validation_obs_zero)
    validation_labels_zero = np.array(validation_labels_zero)

    validation_obs_nonzero, validation_labels_nonzero = zip(*validation_data_nonzero)
    validation_obs_nonzero = np.array(validation_obs_nonzero)
    validation_labels_nonzero = np.array(validation_labels_nonzero)

    validation_all = validation_data_nonzero + validation_data_zero
    validation_obs, validation_labels = zip(*validation_all)
    validation_obs = np.array(validation_obs)
    validation_labels = np.array(validation_labels)


    print("obs", type(training_obs))
    print("obs", training_obs.shape)
    print('labels', type(training_labels))
    print('labels', training_labels.shape)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = TrexNet()
    reward_net.to(device)
    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward_classifier(reward_net, optimizer, training_obs, training_labels, num_iter, batch_size, args.reward_model_path)

    #restore best reward
    reward_net = TrexNet()
    reward_net.load_state_dict(torch.load(args.reward_model_path))
    reward_net.to(device)
    accuracy = calc_accuracy(reward_net, training_obs, training_labels)
    print("Train Error = ", accuracy)
    val_accuracy = calc_accuracy(reward_net, validation_obs, validation_labels)
    print("Validation Error All = ", val_accuracy)
    val_accuracy_zero = calc_accuracy(reward_net, validation_obs_zero, validation_labels_zero)
    print("Validation Error Zero Reward = ", val_accuracy_zero)
    val_accuracy_nonzero = calc_accuracy(reward_net, validation_obs_nonzero, validation_labels_nonzero)
    print("Validation Error Nonzero Reward= ", val_accuracy_nonzero)

    #TODO:add checkpoints to training process
    #torch.save(reward_net.state_dict(), args.reward_model_path)
