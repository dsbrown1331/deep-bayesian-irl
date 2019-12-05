import numpy as np
import helper
import argparse

#Create tables for NeurIPS workshop paper but with normalized feature counts to see if that helps!
#I'm going to try the L1 and L2 norms

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', help="name of the environment, e.g. 'breakout'")

args = parser.parse_args()
print(args.env_name)
#read in the weights as a 2-d array and the feature counts of the policy
W = helper.get_weightchain_array("../../mcmc_data/" + args.env_name + "_0.txt") #these are all normalized
#print(np.mean(W, axis=0))
eval_policies = ['00025', '00325', '00800', '01450', 'mean', 'map']
name_transform = {'00025':'policy A', '00325':'policy B', '00800':'policy C', '01450':'policy D', 'mean':'mean', 'map':'MAP', 'noop': 'no-op'}
if args.env_name == "enduro":
    eval_policies =['03125', '03425', '03900', '04875', 'mean', 'map']
    name_transform = {'03125':'policy A', '03425':'policy B', '03900':'policy C', '04875':'policy D', 'mean':'mean', 'map':'MAP', 'noop': 'no-op'}
gt_return_list = []
fcount_list = []
return_dist_list = []
print(" policy & mean & 0.05-VaR & ave length & gt & min gt \\\\ \hline")
for eval in eval_policies:
    #print("-"*20)
    #print("eval", eval)
    returns, fcounts = helper.parse_avefcount_array('../../policies/' + args.env_name +'_' + eval + '_fcounts.txt')
    #normalize
    fcounts_l1 = fcounts / np.sum(np.abs(fcounts))
    return_dist = np.dot(W,fcounts_l1)

    print("{} & {:.2E} & {:.2E} & {:.2E} & {:.1f} & {:.0f}  \\\\".format(name_transform[eval], np.mean(return_dist), helper.worst_percentile(return_dist, 0.05), fcounts[-1], np.mean(returns), np.min(returns)))

if args.env_name == "breakout":
    #evaluate the no-op policy
    returns, fcounts = helper.parse_avefcount_array('../../policies/breakout_noop_fcounts.txt')
    #normalize
    fcounts_l1 = fcounts / np.sum(np.abs(fcounts))
    return_dist = np.dot(W,fcounts_l1)

    print("{} & {:.2E} & {:.2E} & {:.2E} & {:.1f} & {:.0f}  \\\\".format("no-op", np.mean(return_dist), helper.worst_percentile(return_dist, 0.05), fcounts[-1], np.mean(returns), np.min(returns)))
