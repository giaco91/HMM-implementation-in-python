import numpy as np
import time
import os
from BW_utility import *

#import test_hmm_package as hmm_pk
# from hmm import *
# from discrete import *
# from gaussian import *
from learnhmm import *


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



#[pi,a,e,z]=init(2,2)
# covar1=np.array([[1,0,0],[0,1,0],[0,0,1]])
# covar2=np.array([[1,0,0],[0,2,0],[0,0,1]])
# covar3=np.array([[1,0,0],[0,1.5,0],[0,0,1]])
# mean1=np.array([0,0,0])
# mean2=np.array([2,0,0])
# mean3=np.array([1,0,1])
# covars=[covar1,covar2]
# means=[mean1,mean2]

z=[]
data_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/day06_b7r16/z_sequences'
for filename in os.listdir(data_path):
    if filename.endswith(".npy"):
        z.append(np.load(os.path.join(data_path, filename)))

len(z)
print(z[0].shape)

# [centroids,clusters,new_distance]=Gaussian_emission(K=2,D=2).k_mean(z,2)
# plt.scatter(clusters[0][:,0],clusters[0][:,1],color='green')
# plt.scatter(clusters[1][:,0],clusters[1][:,1],color='yellow')
# plt.scatter(centroids[:,0],centroids[:,1],color='red')
# plt.show()

hmm_gauss=gaussian.Gaussian_emission(K=5,D=z[0].shape[1])
LL=hmm_gauss.fit(z,8,init='k_mean')
print(LL[-1])
# print(hmm_gauss.e[0].mean)
# print(hmm_gauss.e[1].mean)
# print(hmm_gauss.a)
# print(hmm_gauss.e[0].covar)
# print(hmm_gauss.e[0].covar)
# sample=hmm_gauss.sample_observation(z[0],6)
# print(sample)


# hmm_disc=Discrete_emission(K=3,D=3)
# x=hmm_disc.sample_hiddenpath(10)
# [f_s,c,L]=hmm_disc.forward_scaled(z[0])
# b_s=hmm_disc.backward_scaled(z[0],c)
# sample=hmm_disc.sample_observation(z[0],12)
# [x_ml,ML]=hmm_disc.viterbi(z[0])
# print(x_ml)
# LL=hmm_disc.fit(z,10)
# print(LL)
# sample=hmm_disc.sample_observation(z[0],7)
# print(sample)
















