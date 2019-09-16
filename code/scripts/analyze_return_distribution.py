import numpy as np
import helper

#read in the weights as a 2-d array and the feature counts of the policy
W = helper.get_weightchain_array("../../mcmc_data/breakout_0.txt")
print(np.mean(W, axis=0))
eval_policies = ['00050', '00200', '00400', '00600', 'map', 'mean', 'noop']
gt_return_list = []
fcount_list = []
return_dist_list = []
print(" policy & mean & 0.05-VaR & ave length & gt & min gt")
for eval in eval_policies:
    #print("-"*20)
    #print("eval", eval)
    returns, fcounts = helper.parse_avefcount_array('../../policies/breakout_' + eval + '_fcounts.txt')
    return_dist = np.dot(W,fcounts)

    print(eval,  "&", np.mean(return_dist), '&', helper.worst_percentile(return_dist, 0.05), "&", fcounts[-1], "&", np.mean(returns), "&", np.min(returns))

    gt_return_list.append(returns)
    fcount_list.append(fcounts)
    return_dist_list.append(return_dist)
