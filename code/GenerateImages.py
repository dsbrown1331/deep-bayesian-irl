import argparse
import torch.distributions as tdist
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
            gt_rewards = []
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







def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length):
    #collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    times = []
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
        min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        if ti < tj: #pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            #print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - rand_length + 1)
        else: #ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            #print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - rand_length + 1)
        traj_i = demonstrations[ti][ti_start:ti_start+rand_length:2] #skip everyother framestack to reduce size
        traj_j = demonstrations[tj][tj_start:tj_start+rand_length:2]

        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        if ti > tj:
            label = 0
        else:
            label = 1
        len1 = len(traj_i)
        len2 = len(list(range(ti_start, ti_start+rand_length, 2)))
        if len1 != len2:
            print("---------LENGTH MISMATCH!------")
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        times.append((list(range(ti_start, ti_start+rand_length, 2)), list(range(tj_start, tj_start+rand_length, 2))))

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels, times



class Net(nn.Module):
    def __init__(self, ENCODING_DIMS):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc_mu = nn.Linear(64, ENCODING_DIMS)
        self.fc_var = nn.Linear(64, ENCODING_DIMS)
        self.fc2 = nn.Linear(ENCODING_DIMS, 1)
        self.reconstruct1 = nn.Linear(ENCODING_DIMS, 64)
        self.reconstruct2 = nn.Linear(64, 784)
        self.reconstruct_conv1 = nn.ConvTranspose2d(1, 4, 3, stride=1)
        self.reconstruct_conv2 = nn.ConvTranspose2d(4, 16, 6, stride=1)
        self.reconstruct_conv3 = nn.ConvTranspose2d(16, 16, 7, stride=2)
        self.reconstruct_conv4 = nn.ConvTranspose2d(16, 4, 10, stride=1)
        self.temporal_difference1 = nn.Linear(ENCODING_DIMS*2, ENCODING_DIMS)
        self.temporal_difference2 = nn.Linear(ENCODING_DIMS, 1)
        self.normal = tdist.Normal(0, 1)

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
        #print(x.shape)
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
        x = F.leaky_relu(self.reconstruct_conv4(x))
        #print(x.shape)
        #print("------end decoding--------")
        return x.permute(0, 2, 3, 1)

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, mu1, var1, z1 = self.cum_return(traj_i)
        cum_r_j, abs_r_j, mu2, var2, z2 = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j, z1, z2




# Train the network
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, training_times, num_iter, l1_reg, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    reconstruction_loss = nn.MSELoss()
    temporal_difference_loss = nn.MSELoss()
    
    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs, training_times))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels, training_times = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            times_i, times_j = training_times[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards, z1, z2 = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            decoded1 = reward_network.decode(z1)
            #print("DECODED SHAPE:")
            #print(decoded1.shape)
            #print(decoded1.type())
            #print("TRAJ_I SHAPE:")
            #print(traj_i.shape)
            #print(traj_i.type())
            decoded2 = reward_network.decode(z2)
            reconstruction_loss_1 = reconstruction_loss(decoded1, traj_i)
            reconstruction_loss_2 = reconstruction_loss(decoded2, traj_j)

            t1_i = np.random.randint(0, len(times_i))
            t2_i = np.random.randint(0, len(times_i))
            t1_j = np.random.randint(0, len(times_j))
            t2_j = np.random.randint(0, len(times_j))
            
            est_dt_i = reward_network.estimate_temporal_difference(z1[t1_i].unsqueeze(0), z1[t2_i].unsqueeze(0))
            est_dt_j = reward_network.estimate_temporal_difference(z2[t1_j].unsqueeze(0), z2[t2_j].unsqueeze(0))
            real_dt_i = (times_i[t2_i] - times_i[t1_i])/100.0
            real_dt_j = (times_j[t2_j] - times_j[t1_j])/100.0

            #print("est_dt: " + str(est_dt_i) + ", real_dt: " + str(real_dt_i))
            #print("est_dt: " + str(est_dt_j) + ", real_dt: " + str(real_dt_j))
            dt_loss_i = temporal_difference_loss(est_dt_i, torch.tensor(((real_dt_i,),), dtype=torch.float32))
            dt_loss_j = temporal_difference_loss(est_dt_j, torch.tensor(((real_dt_j,),), dtype=torch.float32))

            trex_loss = loss_criterion(outputs, labels)

            loss = reconstruction_loss_1 + reconstruction_loss_2
            #loss = trex_loss + l1_reg * abs_rewards + reconstruction_loss_1 + reconstruction_loss_2 + dt_loss_i + dt_loss_j
            #TODO add l2 reg

            print("!LOSSDATA " + str(reconstruction_loss_1.data.numpy()) + " " + str(reconstruction_loss_2.data.numpy()) + " " + str(dt_loss_i.data.numpy()) + " " + str(dt_loss_j.data.numpy()) + " " + str(trex_loss.data.numpy()) + " " + str(loss.data.numpy()))

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
            outputs, abs_return, z1, z2 = reward_network.forward(traj_i, traj_j)
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


    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)

    demonstrations, learning_returns, learning_rewards = generate_novice_demos(env, env_name, agent, args.models_dir)
    import os
    os.mkdir("output_images")
    nn = 0
    from torchvision.utils import save_image
    for demo in demonstrations:
        os.mkdir("output_images/" + str(nn))
        fnn = 0
        for framestack in demo:
            print(framestack.shape)
            save_image(
                    torch.from_numpy(framestack).permute(2, 0, 1).reshape(84*4, 84),
                    "output_images/" + str(nn) + "/" + str(fnn) + ".png"
            )
            fnn += 1
        nn += 1
    #sort the demonstrations according to ground truth reward to simulate ranked demos
    """

    demo_lengths = [len(d) for d in demonstrations]
    print("demo lengths", demo_lengths)
    max_snippet_length = min(np.min(demo_lengths), maximum_snippet_length)
    print("max snippet length", max_snippet_length)

    print(len(learning_returns))
    print(len(demonstrations))
    print([a[0] for a in zip(learning_returns, demonstrations)])
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)
    
    training_obs, training_labels, training_times = create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
    print("num_times", len(training_times))
   
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net(encoding_dims)
    reward_net.to(device)
    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_obs, training_labels, training_times, num_iter, l1_reg, args.reward_model_path)
    #save reward network
    torch.save(reward_net.state_dict(), args.reward_model_path)
    
    #print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))

    """
    
