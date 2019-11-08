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
#parser.add_argument('--env_name', help="name of the environment, e.g. 'breakout'")

args = parser.parse_args()


eval_games = ['beamrider', 'breakout', 'enduro', 'seaquest', 'spaceinvaders']
gt_return_list = []
fcount_list = []
return_dist_list = []

for game in eval_games:
    print(game)
    for eval in ["mean", "map"]:
        print(eval)
        #print("eval", eval)
        returns, fcounts = helper.parse_avefcount_array('../../policies/' + game +  '_' + eval + '_fcounts.txt')
        print("{:.0f} & {:.1f}".format(np.min(returns), np.mean(returns)))
