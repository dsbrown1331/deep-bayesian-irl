import numpy as np

def sample_open_interval():
    x = 0
    while x == 0:
        x = np.random.random_sample()
    return x

def sample_from_custom_dist():
    sample = sample_open_interval()
    if sample < 0.5:
        return np.log(2*sample)
    elif sample == 0.5:
        return 0
    else:
        return -np.log(2*(1-sample))

def sample_from_l1_ball(dimensions):
    x = [0] * dimensions
    sabs = 0
    for i in range(dimensions):
        x[i] = sample_from_custom_dist()
        sabs += abs(x[i])
    #y = np.random.exponential(scale=1.0)
    denom = sabs# + y
    for i in range(dimensions):
        x[i] /= denom
    return x

if __name__ == "__main__":
    for x in range(227000):
        data = sample_from_l1_ball(30)
        print("\t".join(str(x) for x in data))
        #print(f"({data[0]}, {data[1]}), ", end="")
    """
    while True:
        dimensions = int(input("> "))
        sample = sample_from_l1_ball(dimensions)
        print(sample)
        print("sum abs: " + str(sum(map(abs, sample))))
    """
