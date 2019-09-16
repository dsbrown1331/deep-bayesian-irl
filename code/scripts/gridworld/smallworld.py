#goal is to run MCMC on small world from my Machine Teaching paper and see what the mixing looks like

#TODO: run experiment where I increase the number of trajectories with preferences

#interesting! we probably need an inverse temperature on the confidence in the rankings! This seems to help the chain mix better and not accept everythin.
#it is also a function of the data, the more data and preferences the less likely I am to accept something bad. This is cool! The size of data set influences
#the width of the posterior, as it should.

import numpy as np
import copy
import matplotlib.pyplot as plt



def loglikelihood(w, beta, fcounts, prefs):
    #fcountA = np.array([1 + 0.9**2, 0.9])#np.array([2.0, 1.0])
    #fcountB = np.array([1 + 0.9 + 0.9**2, 0.0])#np.array([3.0, 0.0])
    #fcountC = np.array([1 + 0.9 + 0.9**2 + 0.9**3 + 0.9**4, 0.0])#np.array([5.0, 0.0])




    loglike = 0.0
    for i,j in prefs:
        #prefer j over i
        loglike += np.log(np.exp(beta * np.dot(w,fcounts[j]))/(np.exp(beta* np.dot(w,fcounts[i])) + np.exp(beta*np.dot(w,fcounts[j]))))
    return loglike

fcounts = [
np.array([2.0, 1.0]),  #0
np.array([3.0, 0.0]),  #1
np.array([5.0, 1.0]),  #2
#np.array([3.0, 1.0]),  #3
#np.array([6.0, 0.0])   #4
]

wstar = np.array([-0.1, -0.9])
#automatically generate preferences
prefs = []
for i in range(len(fcounts)):
    for j in range(i+1, len(fcounts)):
        return_i = np.dot(fcounts[i], wstar)
        return_j = np.dot(fcounts[j], wstar)
        if return_i > return_j:
            prefs.append((j,i))
        elif return_i < return_j:
            prefs.append((i,j))

print(prefs)
input("Continue")
# print(loglikelihood(np.array([1,0])))
# print(loglikelihood(np.array([0,1])))
# print(loglikelihood(np.array([1,1])))
# print(loglikelihood(np.array([-1,0])))
# print(loglikelihood(np.array([0,-1])))
# print(loglikelihood(np.array([-1,-1])))
# print(loglikelihood(np.array([1,-1])))
debug = False
N = 2000
beta = 10.0
prop_width = 1.0
#start with random weight
w = np.random.randn(2)
#normalize
w = w / np.sum(np.abs(w))
print(w)
cur_w = copy.deepcopy(w)
cur_ll = loglikelihood(w, beta, fcounts, prefs)
map_w = copy.deepcopy(w)
map_ll = cur_ll
#run mcmc
chain = []
num_accept = 0
for i in range(N):
    if i % 1000 == 0:
        print(i)
    if debug:
        print()
        print(i)
        print("current", cur_w)
        print("current likelihood", cur_ll)
    #generate proposal
    prop_w = cur_w + np.random.randn(len(cur_w))*prop_width
    #normalize
    prop_w = prop_w / np.sum(np.abs(prop_w))
    if debug:
        print("proposal", prop_w)
    #get proposal log_likelihood
    prop_ll = loglikelihood(prop_w, beta, fcounts, prefs)
    if debug:
        print("prop likelihood", prop_ll)
    #check likelihood ratio
    if prop_ll > cur_ll:
        if debug:
            print("accept")
        #update MAP
        if prop_ll > map_ll:
            map_ll = prop_ll
            map_w = copy.deepcopy(prop_w)
            print("new map", map_w, "ll", map_ll)
        cur_w = copy.deepcopy(prop_w)
        cur_ll = prop_ll
        chain.append(prop_w)
        num_accept += 1
    else: #check for probabilistic reject
        r = np.random.rand()
        ll_ratio = np.exp(prop_ll - cur_ll)
        if debug:
            print(ll_ratio, r)
        if r < ll_ratio:
            if debug:
                print("probabilistic accept")
            cur_w = copy.deepcopy(prop_w)
            cur_ll = prop_ll
            chain.append(prop_w)
            num_accept += 1
        else:
            if debug:
                print("reject")
            #keep old sample
            chain.append(cur_w)
chain = np.array(chain)

print("map", map_w)
print("likelihood", map_ll)
print("mean", np.mean(chain, axis=0))
print("accept / total = {} / {}".format(num_accept, N))
plt.figure(0)
plt.plot(chain[:,0])
plt.title("w0")
plt.figure(1)
plt.plot(chain[:,1])
plt.title("w1")
plt.show()
