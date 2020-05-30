### This code will take in any pretrained network and compute the expected feature counts via Monte Carlo sampling according to the last
### layer of the pretrained network


import os
import sys
import pickle
import gym
import time
import numpy as np
import random
import torch
from run_test import *
#import matplotlib.pylab as plt
import argparse
from StrippedNet import EmbeddingNet
from baselines.common.trex_utils import preprocess
import utils

network_file_loc = "/home/dsbrown/Code/deep-bayesian-irl/pretrained_networks/auxloss/breakout_64_all.params_stripped.params"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_net = EmbeddingNet(64)
state_dict = torch.load(network_file_loc, map_location=device)
print(state_dict.keys())
print(state_dict['fc2.bias'])
feature_net.load_state_dict(torch.load(network_file_loc, map_location=device))
