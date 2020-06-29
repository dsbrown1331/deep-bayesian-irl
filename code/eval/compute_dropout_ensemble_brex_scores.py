import numpy as np
import sys


postfix1 = '_scratch_cluster_dsbrown_tflogs_mcmc_'
postfix2 = '_checkpoints_43000_evaluation.txt'
for game in ['beamrider', 'breakout', 'enduro', 'seaquest', 'spaceinvaders']:

    filenames = [ game + postfix1 + game + "_64_all" + postfix2,
                  game + postfix1 + game + "_64_all_mean" + postfix2,
                  game + "_ensemble_43000_evaluation.txt"
                  ]
    print(game, end=" & ")
    for fname in filenames:
        f = open(fname)
        returns = []
        for line in f:
            returns.append(float(line))
        if filenames.index(fname) == len(filenames)-1:
            print("{:.1f} ".format(np.mean(returns)), end = "\\\\ \n")
        else:
             print("{:.1f} ".format(np.mean(returns)), end = " & ")
        f.close()
