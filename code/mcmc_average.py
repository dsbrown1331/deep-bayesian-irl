import matplotlib.pyplot as plt
import numpy as np

burn = 1000
skip = 5
reader = open('../mcmc_data/breakout_0.txt')
data = []
for line in reader:
    parsed = line.strip().split(',')
    np_line = []
    for s in parsed[:-1]:
        np_line.append(float(s))
    data.append(np_line)
data = np.array(data)
print(data[burn::skip,:].shape)

#get average across chain
mean_weight = np.mean(data[burn::skip,:], axis = 0)
print(mean_weight)
print(mean_weight.shape)
