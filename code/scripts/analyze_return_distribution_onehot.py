import numpy as np
import helper
import argparse


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', help="name of the environment, e.g. 'breakout'")

alpha = 0.01

args = parser.parse_args()
print(args.env_name)
#read in the weights as a 2-d array and the feature counts of the policy
W, log_lik = helper.get_weightchain_array("../../mcmc_data/" + args.env_name + "_onehot_chain.txt", return_likelihood=True)
print(np.mean(W, axis=0))
eval_policies = ['00025', '00325', '00800', '01450', 'mean', 'map']
name_transform = {'00025':'policy A', '00325':'policy B', '00800':'policy C', '01450':'policy D', 'mean':'mean', 'map':'MAP', 'noop': 'no-op'}
if args.env_name == "enduro":
    eval_policies =['03125', '03425', '03900', '04875', 'mean', 'map']
    name_transform = {'03125':'policy A', '03425':'policy B', '03900':'policy C', '04875':'policy D', 'mean':'mean', 'map':'MAP', 'noop': 'no-op'}
gt_return_list = []
fcount_list = []
return_dist_list = []
print(" policy & mean & " +  str(alpha) + "-VaR & ave length & gt & min gt \\\\ \hline")
for eval in eval_policies:
    #print("-"*20)
    #print("eval", name_transform[eval])
    returns, fcounts = helper.parse_avefcount_array('../../policies/' + args.env_name +'_' + eval + '_fcounts_onehot.txt')
    #print(fcounts)
    return_dist = np.dot(W,fcounts)

    print("{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.0f}  \\\\".format(name_transform[eval], np.mean(return_dist), helper.worst_percentile(return_dist, alpha), np.sum(fcounts), np.mean(returns), np.min(returns)))

    gt_return_list.append(returns)
    fcount_list.append(fcounts)
    return_dist_list.append(return_dist)

if args.env_name == "breakout":
    #I realized that I need to rerun the noop code for the full features. I keep overwriting it.
    #evaluate the no-op policy
    returns, fcounts = helper.parse_avefcount_array('../../policies/breakout_noop_fcounts.txt')

    #normalize
    return_dist = np.dot(W,fcounts)
    return_dist_list.append(return_dist)
    print("{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.0f} \\\\".format("no-op", np.mean(return_dist), helper.worst_percentile(return_dist, alpha), np.sum(fcounts), np.mean(returns), np.min(returns)))


print()
print("fcounts")
for i in range(len(fcount_list)):
    print("{} & {:0.1f} & {:0.1f} & {:0.1f} & {:0.1f} \\\\".format(name_transform[eval_policies[i]], fcount_list[i][0], fcount_list[i][1], fcount_list[i][2], fcount_list[i][2]/max(0.001, fcount_list[i][0])))


#find and print out the MAP and average weights of the distribution from MCMC
print("Mean weights")
print(np.mean(W, axis=0))
print()

max_likelihood = -float('inf')

for i,ll in enumerate(log_lik):
    if ll > max_likelihood:
        max_likelihood = ll
        best_w = W[i]
print("MAP w")
print(best_w)
print()


#debuggin code to try and inspect the MCMC chain to debug the noop policy for breakout
print(W.shape)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(W[:,0], label='-1')
plt.plot(W[:,1], label='0')
plt.plot(W[:,2], label='+1')
plt.legend()

plt.figure()
if args.env_name == "breakout":
    plt.hist([return_dist_list[-2], return_dist_list[-1]],100, label=["MAP", "NOOP"])
else:
    plt.hist([return_dist_list[-2], return_dist_list[-1]],30, label=["Mean", "MAP"])
plt.xlabel("Returns")
plt.ylabel("Frequency")
plt.legend()

plt.figure()
plt.hist(return_dist_list[:4], 30, label=["A", "B", "C", "D"])
plt.legend()
plt.show()
