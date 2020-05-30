import numpy as np
import helper
import argparse
from numpy import genfromtxt


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default='spaceinvaders', type=str, help="name of the environment, e.g. 'breakout'")

args = parser.parse_args()
print(args.env_name)
#read in the weights as a 2-d array and the feature counts of the policy
eval_policies =['.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_everyother_framestacks0.npy_',
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_flee_framestacks0.npy_',
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_hide_framestacks0.npy_',
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_play_framestacks0.npy_',
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_seek_bullets_framestacks0.npy_',
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_shoot_barriers_framestacks0.npy_',
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_shoot_ignore_bullets_framestacks0.npy_',
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_shoot_miss_framestacks0.npy_']#, 'mean', 'map']
name_transform = {'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_everyother_framestacks0.npy_':"every other",
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_flee_framestacks0.npy_': "flee",
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_hide_framestacks0.npy_': "hide",
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_play_framestacks0.npy_':"good",
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_seek_bullets_framestacks0.npy_':"suicidal",
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_shoot_barriers_framestacks0.npy_':"shoot shelters",
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_shoot_ignore_bullets_framestacks0.npy_':"hold'n fire",
'.._.._behavioral_cloning_atari_demos_SpaceInvadersNoFrameskip-v4_shoot_miss_framestacks0.npy_': "miss"}

print(" policy & mean & 0.05-VaR & gt & min gt")
for eval in eval_policies:
    #print("-"*20)
    #print("eval", eval)
    #returns, fcounts = helper.parse_avefcount_array('../../policies/' + args.env_name +'_' + eval + '_fcounts.txt')
    #return_dist = np.dot(W,fcounts)
    write_directory = '../../bayesian_dropout/'
    pred_filename = write_directory + args.env_name + "_" + eval + "pred.txt"
    true_filename = write_directory + args.env_name + "_" + eval + "true.txt"

    return_dist = genfromtxt(pred_filename, delimiter='\n')
    returns = genfromtxt(true_filename, delimiter='\n')
    #let's try and get the data into 5 bins and plot
    # import matplotlib.pyplot as plt
    # plt.plot(return_dist[:20])
    # plt.show()
    #print(returns)

    print("{} & {:.1f} & {:.1f} & {:.1f} & {:.0f}  \\\\".format(name_transform[eval], np.mean(return_dist), helper.worst_percentile(return_dist, 0.05), np.mean(returns), np.min(returns)))
