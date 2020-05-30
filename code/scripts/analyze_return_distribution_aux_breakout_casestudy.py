import numpy as np
import helper
import argparse
import matplotlib.pyplot as plt

#analyze return distributions and VaR for features learned via auxiliary losses


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default='breakout', help="name of the environment, e.g. 'breakout'")
parser.add_argument('--mcmc_file', help="name of mcmc file for chain")
parser.add_argument('--alpha', type=float, help="value of alpha-VaR, e.g. alpha = 0.05")
parser.add_argument('--identifier', help="keyword to find correct fcount files")
parser.add_argument('--plot', action='store_true')
parser.add_argument('--noop', action='store_true')
args = parser.parse_args()


print("alpha:", args.alpha)

print(args.env_name)
#read in the weights as a 2-d array and the feature counts of the policy
W, likelihood = helper.get_weightchain_array(args.mcmc_file, burn=5000, skip=20, return_likelihood=True, preburn_until_accept=True)
print("make sure that I actually accepted a value")
print(np.sum(likelihood == -float('inf')))

W = W[likelihood != -float('inf')]
#print(np.mean(W, axis=0))
eval_policies = ['00025', '00325', '00800', '01450', 'mean', 'map', 'no-op']
name_transform = {'00025':'A', '00325':'B', '00800':'C', '01450':'D', 'mean':'Mean', 'map':'MAP', 'no-op': 'NoOp'}
if args.env_name == "enduro":
    eval_policies =['03125', '03425', '03900', '04875', 'mean', 'map', 'no-op']
    name_transform = {'03125':'A', '03425':'B', '03900':'C', '04875':'D', 'mean':'Mean', 'map':'MAP', 'no-op': 'No-Op'}
gt_return_list = []
fcount_list = []
return_dist_list = []
print(" policy & mean & " + str(args.alpha) + "-VaR  & gt & Length \\\\ \hline")
for eval in eval_policies:
    #print("-"*20)
    #print("eval", eval)
    #if eval == "no-op":
#        fcounts, returns, lengths, all_fcounts = helper.parse_fcount_policy_eval('../../policies/breakout_noopp_64_all.params_stripped.params_fcounts_auxiliary.txt')
#    else:
    fcounts, returns, lengths, all_fcounts = helper.parse_fcount_policy_eval('../../policies/' + args.env_name +'_' + eval + args.identifier + '.params_stripped.params_fcounts_auxiliary.txt')
    return_dist = np.dot(W,fcounts)
    #print(return_dist[:200])
    #input()
    print("{} & {:.1f} & {:.1f}  & {:.1f} & {:.1f}  \\\\".format(name_transform[eval], np.mean(return_dist), helper.worst_percentile(return_dist, args.alpha), np.mean(returns), np.mean(lengths)))




    gt_return_list.append(returns)
    fcount_list.append(fcounts)
    return_dist_list.append(return_dist)

# if args.env_name == "breakout" and args.noop:
#     #evaluate the no-op policy
#     returns, fcounts = helper.parse_avefcount_array('../../policies/breakout_no-op' + args.identifier + '.params_stripped.params_fcounts_auxiliary.txt')
#     noop_returns = returns
#     #normalize
#     return_dist = np.dot(W,fcounts)
#     noop_return_dist = return_dist
#     print("{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.0f} \\\\".format("no-op", np.mean(return_dist), helper.worst_percentile(return_dist, args.alpha), 10*helper.worst_percentile(return_dist, args.alpha) + np.mean(return_dist), 0, np.mean(returns), np.min(returns)))

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
    plt.hist([return_dist_list[eval_policies.index('mean')], return_dist_list[eval_policies.index('map')]],30, label=["Mean", "MAP"])
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.title("predicted")
    plt.legend()

    plt.figure()
    plt.hist([gt_return_list[eval_policies.index('mean')], gt_return_list[eval_policies.index('map')]],30, label=["Mean", "MAP"  ])
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.title("actual")
    plt.legend()

    plt.figure()
    #plt.hist([return_dist_list[-2], return_dist_list[-1], noop_return_dist],30, label=["Mean", "MAP","noop])
    plt.hist([return_dist_list[eval_policies.index('no-op')], return_dist_list[eval_policies.index('map')]],30, label=["no-op", "MAP"])
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.title("predicted")
    plt.legend()



    plt.show()
