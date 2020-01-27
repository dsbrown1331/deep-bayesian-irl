import numpy as np
import helper
import argparse
import matplotlib.pyplot as plt

#analyze return distributions and VaR for features learned via auxiliary losses


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', help="name of the environment, e.g. 'breakout'")
parser.add_argument('--mcmc_file', help="name of mcmc file for chain")
parser.add_argument('--alpha', type=float, help="value of alpha-VaR, e.g. alpha = 0.05")
parser.add_argument('--identifier', help="keyword to find correct fcount files")
args = parser.parse_args()


print("alpha:", args.alpha)

print(args.env_name)
#read in the weights as a 2-d array and the feature counts of the policy
W = helper.get_weightchain_array(args.mcmc_file, burn=5000, skip=100)
print(np.mean(W, axis=0))
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

if args.env_name == "breakout":
    #evaluate the no-op policy
    returns, fcounts = helper.parse_avefcount_array('../../policies/breakout_no-op' + args.identifier + '.params_stripped.params_fcounts_auxiliary.txt')
    noop_returns = returns
    #normalize
    return_dist = np.dot(W,fcounts)
    noop_return_dist = return_dist
    print("{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.0f} \\\\".format("no-op", np.mean(return_dist), helper.worst_percentile(return_dist, args.alpha), 10*helper.worst_percentile(return_dist, args.alpha) + np.mean(return_dist), 0, np.mean(returns), np.min(returns)))


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
