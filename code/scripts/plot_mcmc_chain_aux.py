import numpy as np
import helper
import argparse


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--mcmc_file', help="name of the file with mcmc data")

args = parser.parse_args()

#read in the weights as a 2-d array and the feature counts of the policy
W, log_lik = helper.get_weightchain_array(args.mcmc_file, burn=0, skip=1, return_likelihood=True)
print("average weight", np.mean(W, axis=0))
print(np.argmax(np.mean(W,axis=0)))

max_likelihood = -float('inf')

for i,ll in enumerate(log_lik):
    if ll > max_likelihood:
        max_likelihood = ll
        best_w = W[i]
print("MAP w")
print(best_w)


#debuggin code to try and inspect the MCMC chain to debug the noop policy for breakout
print(W.shape)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(W)
plt.legend()
plt.show()
