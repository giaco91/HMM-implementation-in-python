import numpy as np
import time
from BW_utility import *

# #-----------notation-----------
# #K, number of hidden states
# #N, sequence size
# #D, size of ouput space
# #pi, initial distribution of hidden states
# #a, transition matrice of the MC
# #e, emission probability distributions for each state
# #x, hidden the states
# #z, observations (data)
# #f[n,k]=p(z_1,...,z_n,x_n=k)
# #b[n,k]=p(z_n+1,...,z_N|x_n=k)

# #------------------------------

[pi,a,e,z]=init(3,3)

# print(check_transmat(a))
# print(check_pi(pi))
# print(check_emission(e))
#print(check_forward_backward_consistency(pi,a,e,z[0]))

[pi,a,e,LL]=baumwelch(pi,a,e,z,10)
sample=sample_observation(pi,a,e,z[0],10)
print(sample)
# x=sample_hiddenpath(pi,a,10)
# sample_from_path=sample_from_hiddenpath(x,e)

#[x_ml,ML]=viterbi(pi,a,e,z[0])














