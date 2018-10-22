import numpy as np
import time
from BW_utility import *
from Hmm import *


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# #-----------notation-----------
# #K, number of hidden states
# #N, sequence size
# #D, size of ouput space
# #pi, initial distribution of hidden states
# #a, transition matrice of the MC
# #e, emission probability distributions for each state
# #x, hidden states
# #z, observations (data)
# #f[n,k]=p(z_1,...,z_n,x_n=k)
# #b[n,k]=p(z_n+1,...,z_N|x_n=k)

# #------------------------------

[pi,a,e,z]=init(3,3)
covar1=np.array([[1,0],[0,1]])
covar2=np.array([[1,0],[0,2]])
mean1=np.array([0,0])
mean2=np.array([2,0])
covars=[covar1,covar2]
means=[mean1, mean2]

hmm_gauss=Gaussian_emission(K=2,D=2,means=means, covars=covars)


hmm_disc=Discrete_emission(K=3,D=3,emission_mat=e,init_distr=pi,trans_mat=a)
x=hmm_disc.sample_hiddenpath(10)
[f_s,c,L]=hmm_disc.forward_scaled(z[0])
b_s=hmm_disc.backward_scaled(z[0],c)
sample=hmm_disc.sample_observation(z[0],12)
[x_ml,ML]=hmm_disc.viterbi(z[0])
print(x_ml)
LL=hmm_disc.fit(z,10)
print(LL)
sample=hmm_disc.sample_observation(z[0],7)
print(sample)
















