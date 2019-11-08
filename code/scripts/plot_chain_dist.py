#need to debug the distribution
import argparse
import numpy as np
import helper
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (5, 4),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
#plt.style.use('seaborn-deep')

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', help="name of the environment, e.g. 'breakout'")

args = parser.parse_args()


#read in the weights as a 2-d array and the feature counts of the policy
W = helper.get_weightchain_array("../../mcmc_data/" + args.env_name + "_0.txt")
print(np.mean(W, axis=0))
eval_policies = ['00025', '00325', '00800', '01450']
if args.env_name == "enduro":
    eval_policies = ['03125', '03425', '03900', '04875']
gt_return_list = []
fcount_list = []
return_dist_list = []
print(" policy & mean & 0.05-VaR & ave length & gt & min gt")
for eval in eval_policies:
    #print("-"*20)
    #print("eval", eval)
    returns, fcounts = helper.parse_avefcount_array('../../policies/' + args.env_name +  '_' + eval + '_fcounts.txt')
    return_dist = np.dot(W,fcounts)

    print("{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.0f} & {:.0f} \\\\".format(eval, np.mean(return_dist), helper.worst_percentile(return_dist, 0.01), fcounts[-1], np.mean(returns), np.min(returns), np.std(returns)))

    gt_return_list.append(returns)
    fcount_list.append(fcounts)
    return_dist_list.append(return_dist)
plt.figure(0)
plt.hist(return_dist_list,30, label=['policy A', 'policy B', 'policy C', 'policy D'])
plt.xlabel("Returns")
plt.ylabel("Frequency")
plt.legend( )
plt.tight_layout()
plt.savefig("/home/dsbrown/Dropbox/NeurIPS19SafetyWorkshop/figs/" + args.env_name + "_mcmc_return_dist.png")

plt.figure(1)
plt.hist(gt_return_list, 30, label=['policy A', 'policy B', 'policy C', 'policy D'])
plt.xlabel("Returns")
plt.ylabel("Frequency")
plt.legend( )
plt.tight_layout()
plt.savefig("/home/dsbrown/Dropbox/NeurIPS19SafetyWorkshop/figs/" + args.env_name + "_gt_return_dist.png")

plt.show()
