import numpy as np
import helper
import argparse
from numpy import genfromtxt


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', help="name of the environment, e.g. 'breakout'")
parser.add_argument('--num_ensemble', type=int, default=5, help='number of nets in the ensemble')

args = parser.parse_args()
print(args.env_name)
#read in the weights as a 2-d array and the feature counts of the policy
if args.env_name == "breakout":
    eval_policies = ['00025', '00325', '00800', '01450', 'map', 'noop']
else:
    eval_policies = ['00025', '00325', '00800', '01450', 'noop']
name_transform = {'00025':'policy A', '00325':'policy B', '00800':'policy C', '01450':'policy D', 'mean':'mean', 'map':'MAP', 'noop': 'no-op'}
if args.env_name == "enduro":
    eval_policies =['03125', '03425', '03900', '04875', 'noop']#, 'mean', 'map']
    name_transform = {'03125':'policy A', '03425':'policy B', '03900':'policy C', '04875':'policy D', 'mean':'mean', 'map':'MAP', 'noop': 'no-op'}
gt_return_list = []
fcount_list = []
return_dist_list = []
print(" policy & mean & 0.05-VaR & gt & min gt")


write_directory = '../../ensemble_uncertainty/'
pred_filename = write_directory + args.env_name + "_" + eval_policies[0] + "pred.txt"
#first figure out normalization for data based on first (worse) checkpoint
#let's try and get the data into n bins, one for each ensemble
return_dist = genfromtxt(pred_filename, delimiter='\n')
binned_means_checkpoint_25 = []
for i in range(args.num_ensemble):
    binned_means_checkpoint_25.append(np.mean(return_dist[i::args.num_ensemble]))




for eval in eval_policies:
    #print("-"*20)
    #print("eval", eval)
    #returns, fcounts = helper.parse_avefcount_array('../../policies/' + args.env_name +'_' + eval + '_fcounts.txt')
    #return_dist = np.dot(W,fcounts)
    if eval == "map":
        write_directory = '../../ensemble_uncertainty/'
        eval_fname = ".._learned_policies_breakout_linear_map_"
        pred_filename = write_directory + args.env_name + "_" + eval_fname + "pred.txt"
        true_filename = write_directory + args.env_name + "_" + eval_fname + "true.txt"
    elif eval == "noop":
        write_directory = '../../ensemble_uncertainty/'
        pred_filename = write_directory + args.env_name + "_no_op_pred.txt"
        true_filename = write_directory + args.env_name + "_no_op_true.txt"
    else:
        write_directory = '../../ensemble_uncertainty/'
        pred_filename = write_directory + args.env_name + "_" + eval + "pred.txt"
        true_filename = write_directory + args.env_name + "_" + eval + "true.txt"

    return_dist = genfromtxt(pred_filename, delimiter='\n')
    returns = genfromtxt(true_filename, delimiter='\n')

    #adjust the return_dist by shifting it
    for i in range(args.num_ensemble):
        return_dist[i::args.num_ensemble] -= binned_means_checkpoint_25[i]

    print("{} & {:.1f} & {:.1f} & {:.1f} & {:.0f}  \\\\".format(name_transform[eval], np.mean(return_dist), helper.worst_percentile(return_dist, 0.05), np.mean(returns), np.min(returns)))
