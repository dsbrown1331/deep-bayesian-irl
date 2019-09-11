import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--weight_file_id', help="file with mcmc weights for particular experiment")
args = parser.parse_args()

reader = open('../../mcmc_data/breakout_testing' + args.weight_file_id + '.txt')
data = []
for line in reader:
    parsed = line.strip().split(',')
    np_line = []
    for s in parsed:
        np_line.append(float(s))
    data.append(np_line)
data = np.array(data)
print(data.shape)

#try normalizing after the fact...
row_sums = np.sum(np.abs(data)[:,:-1], axis = 1)
data[:,:-1] = data[:,:-1] / row_sums[:,None] #make broadcasting work
#print(data[:,-2])
#pick a few random indices to see
indices = np.random.choice(len(data[0]), 4, replace=False)
indices.sort()
for i in [0,20,64,65]:#range(60,66):# indices:
    print(data[:,i])
    plt.figure(i)
    plt.title("feature " + str(i))
    plt.plot(data[:,i])
    plt.savefig("../../figs/feature " + str(i) + "chain" + args.weight_file_id + ".png")
plt.show()
