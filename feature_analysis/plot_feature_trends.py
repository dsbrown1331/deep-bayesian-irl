import numpy as np
import matplotlib.pyplot as plt

num_features = 30
env_name = "enduro"


data_all = np.loadtxt(env_name + "_64_all_fcount_demos.txt", delimiter=",")
print(data_all.shape)
plt.figure()
plt.plot(data_all[:,:num_features])
plt.xlabel("Ranked Demonstrations")
plt.ylabel("Feature Counts")
plt.tight_layout()
#plt.title("first {} features trex+self-supervised".format(num_features))
plt.savefig(env_name + "trex+self-supervised.png")

# plt.figure()
# data_all_normed = data_all - np.mean(data_all,axis=0)
# plt.plot(data_all_normed[:,:num_features])
# plt.title("mean normalized for first {} features trex+self-supervised".format(num_features))



data_trex = np.loadtxt(env_name + "_progress_masking_fcount_demos.txt", delimiter=",")
print(data_trex.shape)
plt.figure()
plt.plot(data_trex[:,:num_features])
#plt.title("first {} features trex".format(num_features))
plt.savefig(env_name + "trex.png")

# plt.figure()
# data_trex_normed = data_trex - np.mean(data_trex,axis=0)
# plt.plot(data_trex_normed[:,:num_features])
# plt.title("mean normalized for first {} features trex".format(num_features))

data_aux = np.loadtxt(env_name + "_linear_200_10000_fcount_demos.txt", delimiter=",")
print(data_aux.shape)
plt.figure()
plt.plot(data_aux[:,:num_features])
#plt.title("first {} features self-supervised".format(num_features))
plt.savefig(env_name + "self-supervised.png")

# plt.figure()
# data_aux_normed = data_aux - np.mean(data_aux,axis=0)
# plt.plot(data_aux_normed[:,:num_features])
# plt.title("mean normalized for first {} features self-supervised".format(num_features))

plt.show()
