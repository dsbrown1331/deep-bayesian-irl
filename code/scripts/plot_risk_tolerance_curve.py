import numpy as np
import helper
import argparse
import matplotlib.pyplot as plt
import fnmatch
import os

#plot risk tolerance curve for all the evaluation policies in /scratch/cluster/dsbrown/rl_polices/


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', help="name of the environment, e.g. 'breakout'")
parser.add_argument('--mcmc_file', help="name of mcmc file for chain")
parser.add_argument('--alpha', type=float, help="value of alpha-VaR, e.g. alpha = 0.05")
parser.add_argument('--plot', action='store_true')
parser.add_argument('--noop', action='store_true')
args = parser.parse_args()


print("alpha:", args.alpha)

print(args.env_name)
#read in the weights as a 2-d array and the feature counts of the policy
W = helper.get_weightchain_array(args.mcmc_file, burn=2000, skip=50)
#print(np.mean(W, axis=0))

eval_files = []
for file in os.listdir('/scratch/cluster/dsbrown/rl_policies'):
    if args.env_name in file:
        print(file)
        eval_files.append(file)

for f in eval_files:
        avefcounts, returns, lengths, fcounts = helper.parse_fcount_policy_eval('/scratch/cluster/dsbrown/rl_policies/' + f)
        print(f)
        print(np.mean(returns), np.mean(lengths))

#input()
alphas = np.linspace(0,1.0,21)
best_returns = []
for alpha in alphas:
    print(alpha)
    alpha_vars = []
    for f in eval_files:
        avefcounts, returns, lengths, fcounts = helper.parse_fcount_policy_eval('/scratch/cluster/dsbrown/rl_policies/' + f)
        return_dist = np.dot(W,avefcounts)
        var = helper.worst_percentile(return_dist, alpha)
        alpha_vars.append(var)
    #find the best under this alpha
    print(alpha_vars)
    print(len(alpha_vars))
    best_indx = np.argmax(alpha_vars)
    print("best index", best_indx)
    print("best file", eval_files[best_indx])
    avefcounts, returns, lengths, fcounts = helper.parse_fcount_policy_eval('/scratch/cluster/dsbrown/rl_policies/' + eval_files[best_indx])
    best_returns.append(np.mean(returns))
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
                 # 'figure.figsize': (6, 5),
                          'axes.labelsize': 'xx-large',
                                   'axes.titlesize':'xx-large',
                                            'xtick.labelsize':'xx-large',
                                                     'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
plt.plot(alphas, best_returns, linewidth=3)
plt.xlabel("Risk Tolerance")
plt.ylabel("Expected Return")
plt.tight_layout()

avefcounts, returns, lengths, fcounts = helper.parse_fcount_policy_eval('/scratch/cluster/dsbrown/rl_policies/' + eval_files[best_indx])
plt.hist(np.dot(W,avefcounts))
plt.show()


eval_policies = ['00025', '00325', '00800', '01450', 'mean', 'map']
name_transform = {'00025':'policy A', '00325':'policy B', '00800':'policy C', '01450':'policy D', 'mean':'mean', 'map':'MAP', 'noop': 'no-op'}
if args.env_name == "enduro":
    eval_policies =['03125', '03425', '03900', '04875', 'mean', 'map']
    name_transform = {'03125':'policy A', '03425':'policy B', '03900':'policy C', '04875':'policy D', 'mean':'mean', 'map':'MAP', 'noop': 'no-op'}
gt_return_list = []
fcount_list = []
return_dist_list = []
print(" policy & mean & " + str(args.alpha) + "-VaR & mu+10*Var & ave length & gt & min gt \\\\ \hline")
for eval in eval_policies:
    #print("-"*20)
    #print("eval", eval)
    returns, fcounts = helper.parse_avefcount_array('../../policies/' + args.env_name +'_' + eval + args.identifier + '.params_stripped.params_fcounts_auxiliary.txt')
    #print("num rollouts", len(returns))
    return_dist = np.dot(W,fcounts)

    print("{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.0f}  \\\\".format(name_transform[eval], np.mean(return_dist), helper.worst_percentile(return_dist, args.alpha), 10*helper.worst_percentile(return_dist, args.alpha) + np.mean(return_dist), 0, np.mean(returns), np.min(returns)))

    gt_return_list.append(returns)
    fcount_list.append(fcounts)
    return_dist_list.append(return_dist)

if args.env_name == "breakout" and args.noop:
    #evaluate the no-op policy
    returns, fcounts = helper.parse_avefcount_array('../../policies/breakout_no-op' + args.identifier + '.params_stripped.params_fcounts_auxiliary.txt')
    noop_returns = returns
    #normalize
    return_dist = np.dot(W,fcounts)
    noop_return_dist = return_dist
    print("{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.0f} \\\\".format("no-op", np.mean(return_dist), helper.worst_percentile(return_dist, args.alpha), 10*helper.worst_percentile(return_dist, args.alpha) + np.mean(return_dist), 0, np.mean(returns), np.min(returns)))

if args.plot:

    plt.figure()
    plt.hist(return_dist_list[:4], 30, label=["A", "B", "C", "D"])
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.title("Predicted")
    plt.legend()

    plt.figure()
    plt.hist(gt_return_list[:4], 30, label=["A", "B", "C", "D"])
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.title("actual")
    plt.legend()

    plt.figure()
    #plt.hist([return_dist_list[-2], return_dist_list[-1], noop_return_dist],30, label=["Mean", "MAP","noop])
    plt.hist([return_dist_list[-2], return_dist_list[-1]],30, label=["Mean", "MAP"])
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.title("predicted")
    plt.legend()

    plt.figure()
    plt.hist([gt_return_list[-2], gt_return_list[-1]],30, label=["Mean", "MAP"  ])
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.title("actual")
    plt.legend()



    plt.show()
