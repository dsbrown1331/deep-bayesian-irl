import argparse
import torch.distributions as tdist
# coding: utf-8

# Take length 50 snippets and record the cumulative return for each one. Then determine ground truth labels based on this.

# In[1]:


import sys
import pickle
import gym
from gym import spaces
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from run_test import *
from baselines.common.trex_utils import preprocess
import os

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

        model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
        if env_name == "seaquest":
            model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

        agent.load(model_path)
        episode_count = 1
        for i in range(episode_count):
            done = False
            traj = []
            actions = []
            gt_rewards = []
            r = 0

            ob = env.reset()
            steps = 0
            acc_reward = 0
            #os.mkdir('images/' + str(checkpoint))
            frameno = 0
            while True:
                action = agent.act(ob, r, done)
                ob, r, done, info = env.step(action)
                ob_processed = preprocess(ob, env_name)
                ob_processed = ob_processed[0] #get rid of first dimension ob.shape = (1,84,84,4)
                traj.append(ob_processed)
                actions.append(action[0])
                #save_image(torch.from_numpy(ob_processed).permute(2, 0, 1).reshape(4*84, 84), 'images/' + str(checkpoint) + '/' + str(frameno) + '_action_' + str(action[0]) + '.png')
                frameno += 1

                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]
                if done:
                    print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                    break
            print("traj length", len(traj))
            print("demo length", len(demonstrations))
            demonstrations.append([traj, actions])
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)

    return demonstrations, learning_returns, learning_rewards







def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length):
    #collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    times = []
    actions = []
    num_demos = len(demonstrations)

    #add full trajs (for use on Enduro)
    """
    for n in range(num_trajs):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #create random partial trajs by finding random start frame and random skip frame
        si = np.random.randint(6)
        sj = np.random.randint(6)
        step = np.random.randint(3,7)
        
        traj_i = demonstrations[ti][si::step]  #slice(start,stop,step)
        traj_j = demonstrations[tj][sj::step]
        
        if ti > tj:
            label = 0
        else:
            label = 1
        
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
    """


    #fixed size snippets with progress prior
    for n in range(num_snippets):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #create random snippets
        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length = min(len(demonstrations[ti][0]), len(demonstrations[tj][0]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        if ti < tj: #pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            #print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(demonstrations[tj][0]) - rand_length + 1)
        else: #ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            #print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(tj_start, len(demonstrations[ti][0]) - rand_length + 1)
        traj_i = demonstrations[ti][0][ti_start:ti_start+rand_length:1] #skip everyother framestack to reduce size
        traj_j = demonstrations[tj][0][tj_start:tj_start+rand_length:1]
        traj_i_actions = demonstrations[ti][1][ti_start:ti_start+rand_length:1] #skip everyother framestack to reduce size
        traj_j_actions = demonstrations[tj][1][tj_start:tj_start+rand_length:1]

        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        if ti > tj:
            label = 0
        else:
            label = 1
        len1 = len(traj_i)
        len2 = len(list(range(ti_start, ti_start+rand_length, 1)))
        if len1 != len2:
            print("---------LENGTH MISMATCH!------")
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        times.append((list(range(ti_start, ti_start+rand_length, 1)), list(range(tj_start, tj_start+rand_length, 1))))
        actions.append((traj_i_actions, traj_j_actions))

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels, times, actions



class Net(nn.Module):
    def __init__(self, ENCODING_DIMS):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        intermediate_dimension = min(784, max(64, ENCODING_DIMS*2))
        self.fc1 = nn.Linear(784, intermediate_dimension)
        self.fc_mu = nn.Linear(intermediate_dimension, ENCODING_DIMS)
        self.fc_var = nn.Linear(intermediate_dimension, ENCODING_DIMS)
        self.fc2 = nn.Linear(ENCODING_DIMS, 1)
        self.reconstruct1 = nn.Linear(ENCODING_DIMS, intermediate_dimension)
        self.reconstruct2 = nn.Linear(intermediate_dimension, 784)
        self.reconstruct_conv1 = nn.ConvTranspose2d(1, 4, 3, stride=1)
        self.reconstruct_conv2 = nn.ConvTranspose2d(4, 16, 6, stride=1)
        self.reconstruct_conv3 = nn.ConvTranspose2d(16, 16, 7, stride=2)
        self.reconstruct_conv4 = nn.ConvTranspose2d(16, 4, 10, stride=1)
        self.temporal_difference1 = nn.Linear(ENCODING_DIMS*2, ENCODING_DIMS)
        self.temporal_difference2 = nn.Linear(ENCODING_DIMS, 1)
        self.inverse_dynamics1 = nn.Linear(ENCODING_DIMS*2, ENCODING_DIMS)
        self.inverse_dynamics2 = nn.Linear(ENCODING_DIMS, 6)
        self.forward_dynamics1 = nn.Linear(ENCODING_DIMS + 6, (ENCODING_DIMS + 6) * 2)
        self.forward_dynamics2 = nn.Linear((ENCODING_DIMS + 6) * 2, (ENCODING_DIMS + 6) * 2)
        self.forward_dynamics3 = nn.Linear((ENCODING_DIMS + 6) * 2, ENCODING_DIMS)
        self.normal = tdist.Normal(0, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        print("Intermediate dimension calculated to be: " + str(intermediate_dimension))

    def reparameterize(self, mu, var): #var is actually the log variance
        if self.training:
            std = var.mul(0.5).exp()
            eps = self.normal.sample(mu.shape)
            return eps.mul(std).add(mu)
        else:
            return mu


    def cum_return(self, traj):
        #print("input shape of trajectory:")
        #print(traj.shape)
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
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        z = self.reparameterize(mu, var)

        r = self.fc2(z)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards, mu, var, z

    def estimate_temporal_difference(self, z1, z2):
        x = F.leaky_relu(self.temporal_difference1(torch.cat((z1, z2), 1)))
        x = self.temporal_difference2(x)
        return x

    def forward_dynamics(self, z1, actions):
        x = torch.cat((z1, actions), dim=1)
        x = F.leaky_relu(self.forward_dynamics1(x))
        x = F.leaky_relu(self.forward_dynamics2(x))
        x = self.forward_dynamics3(x)
        return x

    def estimate_inverse_dynamics(self, z1, z2):
        concatenation = torch.cat((z1, z2), 1)
        x = F.leaky_relu(self.inverse_dynamics1(concatenation))
        x = F.leaky_relu(self.inverse_dynamics2(x))
        return x

    def decode(self, encoding):
        x = F.leaky_relu(self.reconstruct1(encoding))
        x = F.leaky_relu(self.reconstruct2(x))
        x = x.view(-1, 1, 28, 28)
        #print("------decoding--------")
        #print(x.shape)
        x = F.leaky_relu(self.reconstruct_conv1(x))
        #print(x.shape)
        x = F.leaky_relu(self.reconstruct_conv2(x))
        #print(x.shape)
        x = F.leaky_relu(self.reconstruct_conv3(x))
        #print(x.shape)
        x = self.sigmoid(self.reconstruct_conv4(x))
        #print(x.shape)
        #print("------end decoding--------")
        return x.permute(0, 2, 3, 1)

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, mu1, var1, z1 = self.cum_return(traj_i)
        cum_r_j, abs_r_j, mu2, var2, z2 = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j, z1, z2, mu1, mu2, var1, var2



def reconstruction_loss(decoded, target, mu, logvar):
    num_elements = decoded.numel()
    target_num_elements = decoded.numel()
    if num_elements != target_num_elements:
        print("ELEMENT SIZE MISMATCH IN RECONSTRUCTION")
        sys.exit()
    bce = F.binary_cross_entropy(decoded, target)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld /= num_elements
    print("bce: " + str(bce) + " kld: " + str(kld))
    return bce + kld





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
            outputs, abs_return, z1, z2, _, _, _, _ = reward_network.forward(traj_i, traj_j)
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
    parser.add_argument('--models_dir', default = ".", help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    parser.add_argument('--num_trajs', default = 0, type=int, help="number of downsampled full trajectories")
    parser.add_argument('--num_snippets', default = 6000, type = int, help = "number of short subtrajectories to sample")
    parser.add_argument('--encoding_dims', default = 30, type = int, help = "number of dimensions in the latent space")

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
    num_trajs =  args.num_trajs
    num_snippets = args.num_snippets
    encoding_dims = args.encoding_dims
    min_snippet_length = 50 #min length of trajectory for training comparison
    maximum_snippet_length = 100

    lr = 0.00005
    weight_decay = 0.0
    num_iter = 5 #num times through training data
    l1_reg=0.0
    stochastic = True

    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })

    if env.action_space != spaces.Discrete(6):
        print("Wrong size of action space! Should be discrete of size 6 but is " + str(env.action_space))
        sys.exit()


    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)

    demonstrations, learning_returns, learning_rewards = generate_novice_demos(env, env_name, agent, args.models_dir)

    #sort the demonstrations according to ground truth reward to simulate ranked demos

    demo_lengths = [len(d[0]) for d in demonstrations]
    demo_action_lengths = [len(d[1]) for d in demonstrations]
    for i in range(len(demo_lengths)):
        assert(demo_lengths[i] == demo_action_lengths[i])
    print("demo lengths", demo_lengths)
    max_snippet_length = min(np.min(demo_lengths), maximum_snippet_length)
    print("max snippet length", max_snippet_length)

    print(len(learning_returns))
    print(len(demonstrations))
    print([a[0] for a in zip(learning_returns, demonstrations)])
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)
    
    training_obs, training_labels, training_times, training_actions = create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
    print("num_times", len(training_times))
    print("num_actions", len(training_actions))
   
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net(encoding_dims)
    sd = reward_net.state_dict()
    sd.update(torch.load(args.reward_model_path))
    reward_net.load_state_dict(sd)
    reward_net.to(device)
    reward_net.eval()
    import torch.optim as optim

    #print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj[0]) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))
