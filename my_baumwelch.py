import numpy as np
import time
from BW_utility import *
from Hmm import *

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

[pi,a,e,z]=init(2,3)
hmm_disc=Discrete_emission(K=2,D=3,emission_mat=e,init_distr=pi,trans_mat=a)
x=hmm_disc.sample_hiddenpath(10)
[f_s,c,L]=hmm_disc.forward_scaled(z[0])
b_s=hmm_disc.backward_scaled(z[0],c)
sample=hmm_disc.sample_observation(z[0],12)
[x_ml,ML]=hmm_disc.viterbi(z[0])
print(x_ml)


# print(check_transmat(a))
# print(check_pi(pi))
# print(check_emission(e))
#print(check_forward_backward_consistency(pi,a,e,z[0]))

# [pi,a,e,LL]=baumwelch(pi,a,e,z,10)
# sample=sample_observation(pi,a,e,z[0],10)
# print(sample)
# print(LL)
#print(sample)
# x=sample_hiddenpath(pi,a,10)
# sample_from_path=sample_from_hiddenpath(x,e)

#[x_ml,ML]=viterbi(pi,a,e,z[0])














