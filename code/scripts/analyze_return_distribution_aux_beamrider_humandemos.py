import numpy as np
import helper
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class EmbeddingNet(nn.Module):
    def __init__(self, ENCODING_DIMS):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ENCODING_DIMS = ENCODING_DIMS
        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 16, 3, stride=1)

        # This is the width of the layer between the convolved framestack
        # and the actual latent space. Scales with self.ENCODING_DIMS
        intermediate_dimension = min(784, max(64, self.ENCODING_DIMS*2))

        # Brings the convolved frame down to intermediate dimension just
        # before being sent to latent space
        self.fc1 = nn.Linear(784, intermediate_dimension)

        # This brings from intermediate dimension to latent space. Named mu
        # because in the full network it includes a var also, to sample for
        # the autoencoder
        self.fc_mu = nn.Linear(intermediate_dimension, self.ENCODING_DIMS)

        # This is the actual T-REX layer; linear comb. from self.ENCODING_DIMS
        self.fc2 = nn.Linear(self.ENCODING_DIMS, 1, bias=False)


#analyze return distributions and VaR for features learned via auxiliary losses


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', help="name of the environment, e.g. 'breakout'")
parser.add_argument('--mcmc_file', help="name of mcmc file for chain")
parser.add_argument('--alpha', type=float, help="value of alpha-VaR, e.g. alpha = 0.05")


args = parser.parse_args()

best_reward = EmbeddingNet(64)
#best_reward.fc2 = nn.Linear(num_features, 1, bias=False)
best_reward.load_state_dict(torch.load("/home/dsbrown/Code/deep-bayesian-irl/mcmc_data/beamrider_64_all_map.params", map_location=device))
map_weights = best_reward.fc2.weight.data.cpu().numpy()

print("alpha:", args.alpha)

print(args.env_name)
#read in the weights as a 2-d array and the feature counts of the policy
W, likelihood = helper.get_weightchain_array(args.mcmc_file, burn=5000, skip=20, return_likelihood=True, preburn_until_accept=True)

gt_return_list = []
fcount_list = []
return_dist_list = []
#print(np.mean(W, axis=0))
print("policy & map & posterior ave & best-case & worst-case & ground truth & ave length\\\\")
#eval_policies = ['beamrider_rl_fcounts.txt', 'beamrider_brex_fcounts.txt', 'beamrider_reward_hack_fcounts.txt']
eval_policies = ['human_good.txt', 'human_bad.txt', 'human_suicidal.txt', 'human_adversarial.txt']
#eval_policies = ['beamrider_rl_fcounts_30000.txt', 'beamrider_brex_fcounts_30000.txt', 'beamrider_reward_hack_fcount_30000.txt']
for eval in eval_policies:
    fcounts, returns, lengths, all_fcounts = helper.parse_fcount_policy_eval('/home/dsbrown/Code/deep-bayesian-irl/beamrider_eval_policies/' + eval)
    return_dist = np.dot(W,fcounts)
    map_return = np.dot(map_weights, fcounts)[0]
    #print(map_return)
    #print(return_dist[:200])
    #input()
    #print("{} & {:.1f}  & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} \\\\".format(eval, map_return, np.mean(return_dist), helper.worst_percentile(return_dist, 1-args.alpha), helper.worst_percentile(return_dist, args.alpha), np.mean(returns), np.mean(lengths)))
    print("{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} \\\\".format(eval, np.mean(return_dist),helper.worst_percentile(return_dist, args.alpha), np.mean(returns), np.mean(lengths)))




    gt_return_list.append(returns)
    fcount_list.append(fcounts)
    return_dist_list.append(return_dist)
