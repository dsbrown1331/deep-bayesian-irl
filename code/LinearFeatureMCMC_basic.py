import argparse

# Use hand tuned feature count vectors for testing the MCMC convergence and distributions.
# Doesn't actually use any kind of real domain.


import numpy as np
import random
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from nnet import Net




def calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_cnts):
    '''use (i,j) indices and precomputed feature counts to do faster pairwise ranking loss'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    #print(device)
    #don't need any gradients
    with torch.no_grad():

        #do matrix multiply with last layer of network and the demo_cnts
        #print(list(reward_net.fc2.parameters()))

        #print(linear)
        #print(bias)
        weights = torch.from_numpy(last_layer) #append bias and weights from last fc layer together
        demo_cnts_tensor = torch.from_numpy(demo_cnts)
        #print('weights',weights)
        #print('demo_cnts', demo_cnts)
        demo_returns = torch.mv(demo_cnts_tensor, weights)


        loss_criterion = nn.CrossEntropyLoss(reduction='sum') #sum up losses
        cum_log_likelihood = 0.0
        outputs = torch.zeros(len(pairwise_prefs),2) #each row is a new pair of returns
        for p, ppref in enumerate(pairwise_prefs):
            i,j = ppref
            outputs[p,:] = torch.tensor([demo_returns[i], demo_returns[j]])
        labels = torch.ones(len(pairwise_prefs)).long()

        #outputs = outputs.unsqueeze(0)
        #print(outputs)
        #print(labels)
        cum_log_likelihood = -loss_criterion(outputs, labels)
            #if labels == 0:
            #    log_likelihood = torch.log(return_i/(return_i + return_j))
            #else:
            #    log_likelihood = torch.log(return_j/(return_i + return_j))
            #print("ll",log_likelihood)
            #cum_log_likelihood += log_likelihood
    return cum_log_likelihood.item()


def generate_clipped_feature_counts(demos, rewards, num_features):
    feature_cnts = np.zeros((len(demos), num_features)) #+1 for bias term
    for i in range(len(demos)):
        #count up how many times it was a -1, 0, or +1 reward
        minus_cnt = 0
        zero_cnt = 0
        plus_cnt = 0
        for r in rewards[i]:
            if np.sign(r) == -1:
                minus_cnt += 1
            elif np.sign(r) == 0:
                zero_cnt += 1
            elif np.sign(r) == 1:
                plus_cnt += 1

        #print(np.array([minus_cnt, zero_cnt, plus_cnt, len(demos[i])]))
        feature_cnts[i,:] = np.array([minus_cnt, zero_cnt, plus_cnt])  #append +1 for each timestep for bias weight
    return feature_cnts



def mcmc_map_search(pairwise_prefs, demo_cnts, num_steps, step_stdev, confidence):
    '''run metropolis hastings MCMC and record weights in chain'''





    last_layer = np.random.randn(len(demo_cnts[0]))

    #normalize the weight vector to have unit 2-norm
    last_layer = last_layer / np.linalg.norm(last_layer)
    #last_layer = euclidean_proj_l1ball(last_layer)

    #import time
    #start_t = time.time()
    #starting_loglik = calc_pairwise_ranking_loss(reward_net, demo_pairs, preference_labels)
    #end_t = time.time()
    #print("slow likelihood", starting_loglik, "time", 1000*(end_t - start_t))
    #start_t = time.time()
    starting_loglik = calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_cnts)
    #end_t = time.time()
    #print("new fast? likelihood", new_starting_loglik, "time", 1000*(end_t - start_t))
    #print(bunnY)

    map_loglik = starting_loglik
    map_reward = copy.deepcopy(last_layer)

    cur_reward = copy.deepcopy(last_layer)
    cur_loglik = starting_loglik

    #print(cur_reward)

    reject_cnt = 0
    accept_cnt = 0
    chain = []
    log_liks = []
    for i in range(num_steps):
        #take a proposal step
        proposal_reward = cur_reward + np.random.normal(size=cur_reward.shape) * step_stdev #add random noise to weights of last layer

        #project
        proposal_reward = proposal_reward / np.linalg.norm(proposal_reward)

        #calculate prob of proposal
        prop_loglik = calc_linearized_pairwise_ranking_loss(proposal_reward, pairwise_prefs, demo_cnts)
        if prop_loglik > cur_loglik:
            #accept always
            accept_cnt += 1
            cur_reward = copy.deepcopy(proposal_reward)
            cur_loglik = prop_loglik

            #check if this is best so far
            if prop_loglik > map_loglik:
                map_loglik = prop_loglik
                map_reward = copy.deepcopy(proposal_reward)
                print()
                print("step", i)

                print("updating map loglikelihood to ", prop_loglik)
                print("MAP reward so far", map_reward)
        else:
            #accept with prob exp(prop_loglik - cur_loglik)
            if np.random.rand() < np.exp(prop_loglik - cur_loglik):
                accept_cnt += 1
                cur_reward = copy.deepcopy(proposal_reward)
                cur_loglik = prop_loglik
            else:
                #reject and stick with cur_reward
                reject_cnt += 1
        chain.append(cur_reward)
        log_liks.append(cur_loglik)

    print("num rejects", reject_cnt)
    print("num accepts", accept_cnt)
    return map_reward, np.array(chain), np.array(log_liks)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--num_mcmc_steps', default=2000, type = int, help="number of proposals to generate for MCMC")
    parser.add_argument('--mcmc_step_size', default = 0.2, type=float, help="proposal step is gaussian with zero mean and mcmc_step_size stdev")
    parser.add_argument('--confidence', default=1, type=int, help='confidence in rankings, affects the beta parameter in the softmax')

    args = parser.parse_args()

    print("Hand crafted feature expectations")
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)



    num_features = 3

    #TODO: need to manually specify feature count vectors and true weights below
    demo_cnts = np.array([[  48., 9.,   60.],
                          [  46., 10.,   62.],
                          [  55., 11.,   64.],
                          [  47., 12.,   68.],
                          [  48., 13.,   73.],
                          [  44., 14.,   75.],
                          [  40., 15.,   75.],
                          [  52., 16.,   90.],
                          [  43., 16.,   96.],
                          [  55., 17.,  118.],
                          [  47., 19.,  140.],
                          [  49., 21.,  152.]])

    true_weights = np.array([0,0,1])



    sorted_returns = np.dot(demo_cnts, true_weights)
    print("returns", sorted_returns)


    #just need index tuples (i,j) denoting j is preferred to i. Assuming all pairwise prefs for now
    #check if really better, there might be ties!
    pairwise_prefs = []
    for i in range(len(sorted_returns)):
        for j in range(i+1, len(sorted_returns)):
            if sorted_returns[i] < sorted_returns[j]:
                pairwise_prefs.append((i,j))
            else: # they are equal
                print("equal prefs", i, j, sorted_returns[i], sorted_returns[j])
                pairwise_prefs.append((i,j))
                pairwise_prefs.append((j,i))
    print("pairwise prefs (left preferred to right elements")
    print(pairwise_prefs)

    #Run mcmc
    best_reward_lastlayer,chain,log_liks = mcmc_map_search(pairwise_prefs, demo_cnts, args.num_mcmc_steps, args.mcmc_step_size, args.confidence)

    print("best reward weights", best_reward_lastlayer)
    print("predictions over ranked demos vs. ground truth returns")
    pred_returns = np.dot(demo_cnts, best_reward_lastlayer)
    #print(pred_returns)
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(log_liks)
    plt.title("log likelihoods")
    plt.figure()
    for i in range(num_features):
        plt.plot(chain[:,i],label='feature ' + str(i))
    plt.title("features")
    plt.legend()
    plt.show()
    #print(demo_cnts.shape)

    #add random
