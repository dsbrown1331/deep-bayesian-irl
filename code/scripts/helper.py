import numpy as np

def get_weightchain_array(mcmc_chain_filename, burn=1000, skip=5, return_likelihood=False):
    #load the mean of the MCMC chain
    reader = open(mcmc_chain_filename)
    data = []
    likelihood = []
    for line in reader:
        parsed = line.strip().split(',')
        np_line = []
        for s in parsed[:-1]: #don't get last element since it's the likelihood
            np_line.append(float(s))
        likelihood.append(float(parsed[-1]))
        data.append(np_line)
    data = np.array(data)
    data = data[burn::skip,:]
    likelihood = np.array(likelihood)
    likelihood = likelihood[burn::skip]
    print("chain shape", data.shape)
    reader.close()
    if not return_likelihood:
        return data
    else:
        return data, likelihood

def parse_avefcount_array(fcount_file):
    '''returns a list of returns and a numpy array of ave feature counts'''
    reader = open(fcount_file)
    weights = []
    returns = []
    for i,line in enumerate(reader):
        if i == 0: #read in the np_weights
            parsed = line.strip().split(',')
            for w in parsed:
                weights.append(float(w))
        elif i == 1: #read in the returns
            parsed = line.strip().split(',')
            for r in parsed:
                returns.append(float(r))
    return returns, weights

def worst_percentile(_sequence, alpha):
    sorted_seq = sorted(_sequence)
    #find alpha percentile
    alpha_indx = int(np.floor(alpha * len(_sequence)))
    return sorted_seq[alpha_indx]
