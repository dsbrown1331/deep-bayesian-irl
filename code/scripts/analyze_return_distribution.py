import numpy as np
import helper

#read in the weights as a 2-d array and the feature counts of the policy
W = helper.get_weightchain_array("../../mcmc_data/breakout_0.txt")
eval_policies = ['00050', '00200', '00400', '00600', 'map', 'noop']
gt_return_list = []
fcount_list = []
return_dist_list = []
for eval in eval_policies:
    print("-"*20)
    print("eval", eval)
    returns, fcounts = helper.parse_avefcount_array('../../policies/breakout_' + eval + '_fcounts.txt')
    return_dist = np.dot(W,fcounts)

    print("mean", np.mean(return_dist), 'median', np.median(return_dist), 'stdev', np.std(return_dist), '0.05 Var', helper.worst_percentile(return_dist, 0.05), "ave length", fcounts[-1], "gt", np.mean(returns))

    gt_return_list.append(returns)
    fcount_list.append(fcounts)
    return_dist_list.append(return_dist)
