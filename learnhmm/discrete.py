#------------------------------------------------------------------------
#Author: Sandro Giacomuzzi
#Part of my Master Thesis at ETHZ
#Written at: 1.11.2018
#------------------------------------------------------------------------

import numpy as np
#from distributions import *
import time
from learnhmm.hmm import Hmm

#help(learnhmm)

#from hmm import Hmm

#-----------notation-----------
#K, number of hidden states
#N, sequence size
#D, size of ouput space
#pi, initial distribution of hidden states
#a, transition matrice of the MC
#e, emission probability distributions for each state
#x, hidden states
#z, observations (data)
#f[n,k]=p(z_1,...,z_n,x_n=k)
#b[n,k]=p(z_n+1,...,z_N|x_n=k)



class Discrete_emission(Hmm):

    def __init__(self,K,D,init_distr=None,trans_mat=None,emission_mat=None,print_every=5):
        #---preprocess input-----
        init_distr,trans_mat=self.check_input(K,init_distr,trans_mat,print_every)
        if emission_mat is None:
            #emission_mat=np.zeros((K,D))
            emission_mat=[]
            for k in range(0,K):
                #emission_mat[k,:]=self.get_rp_vector(D)
                emission_mat.append(self.get_rp_vector(D)) 
        else:
            #if emission_mat.shape[0]!=K or emission_mat.shape[1]!=D:
            if len(emission_mat)!=K or emission_mat[0].shape[0]!=D:
                raise ValueError('The shape of the transition matrix must be (K,D)!')
            if not self.check_emission_mat(emission_mat):
                raise ValueError('The given emission matrix is not a proper probability table!')
        #------------------
        Hmm.__init__(self,'discrete',K,init_distr,trans_mat,print_every)#init from super class HMM
        #the emissions and its properties are subclass specific
        self.e=emission_mat
        self.D=D

    def one_hot_decoding(self,z):
        #shape of z: (n,o)
        #assignes to each one-hot-vector an integer value
        z_shape=z.shape
        N=z.shape[0]
        D=z.shape[1]
        z_decoded=np.zeros(N)
        for i in range(0,N):
            k=0
            for j in range(0,D):
                if z[i,j]==1:
                    k=j
            z_decoded[i]=k
        return z_decoded.astype(int)

    def m_step_emission(self,g_i,z):
        S=len(z)
        occurence_s=np.zeros(self.K)
        g_N_sum=np.zeros(self.K)
        
        for s in range(0,S):
            g_N_sum+=np.sum(g_i[s],axis=0)    
        for d in range(0,self.D):
            occurence_s=0
            for s in range(0,S):
                occurence_s+=np.sum(g_i[s]*np.expand_dims(z[s][:,d], axis=1),axis=0)
            #self.e[:,d]=occurence_s/g_N_sum
            helper=occurence_s/g_N_sum
            for i in range(0,self.K):
                self.e[i][d]=helper[i]

    def one_hot_encoding(self,z):
        N=z.shape[0]
        z_encoded=np.zeros((N,D)).astype(int)
        for i in range(0,N):
            z_encoded[i,z[i]]=1
        return z_encoded


    def check_emission_mat(self,e):
        #this funktion checks a to be a allowed emission
        #i.e. that it is a valid probability dirstibution for each k
        # for k in range(0,e.shape[0]):
        for k in range(0,len(e)):
            if not self.check_probability_vec(e[k]):
                return False
        return True

    def fit(self,z,n_iter):
        return self.baumwelch(z,n_iter)



    def sample_observation(self,z,L):
        #pi,a,e, the model parameters
        #z, an initial seed of observations
        #L, the number of predicted points
        #returns an oversvation sample

        N=z.shape[0] #size of sequence
        z_sampled=np.zeros((N+L,self.D))
        z_sampled[0:N,:]=z
        pred_distr=np.zeros(self.D) #pred_distr[k]=p(z_N+1=k|z)
        pred_hidden=np.zeros(self.K) #pred_hidden[k]=p(z_N+1=k|x^N) 
        
        [f_s,c,Likelihood]=self.forward_scaled(z_sampled[0:N,:])
        f_s=f_s[N-1,:]
        for s in range(0,L):
            pred_distr*=0
            pred_hidden*=0
            for k in range(0,self.K):
                for l in range(0,self.K):
                    pred_hidden[k]+=self.a[l,k]*f_s[l]
                for d in range(0,self.D):
                    #pred_distr[d]+=self.e[k,d]*pred_hidden[k]
                    pred_distr[d]+=self.e[k][d]*pred_hidden[k]
            z_sampled[N+s,np.argmax(pred_distr)]=1 #ML-sampling
            [f_s,c]=self.scaled_forward_recursion_step(f_s,z_sampled[N+s,:])
        return z_sampled

    def viterbi(self,z):
        #pi, initial distribution of hidden states
        #a, transition matrice of the MC
        #e, emission probability distributions for each state
        #z, observations (data)
        #This function returns the ML hidden path.
        #Furthermore it returns the ML joint p(x^n,z^n). 

        N=z.shape[0] #size of sequence
        z=self.one_hot_decoding(z)

        #Denote: 
        #d[t,l]=max_x^{t-1}p(x^{t-1},x_t=l,z^n)
        #T[n,l]=argmax_k{d[n-1,k]*a[k,l]}, the tracker
        #x_ml = argmax_x^n{p(x^n,z^n)}

        d=np.zeros((N,self.K))
        T=np.zeros((N,self.K)).astype(int)#note: T[0,:] stays unused, but I still allocate it for convenience
        x_ml=np.zeros((N)).astype(int)

        #----forward to find ML------
        #init
        for k in range(0,self.K):
            #d[0,k]=self.pi[k]*self.e[k,z[0]]
            d[0,k]=self.pi[k]*self.e[k][z[0]]

        #recursion
        for n in range(1,N):
            for l in range(0,self.K):
                #d[n,l]=np.amax(d[n-1,:]*self.a[:,l])*self.e[l,z[n]]
                d[n,l]=np.amax(d[n-1,:]*self.a[:,l])*self.e[l][z[n]]
                T[n,l]=np.argmax(d[n-1,:]*self.a[:,l])
        #ML
        #ML=np.amax(d[N-1,:]*self.e[:,z[N-1]])
        ML=d[N-1,0]*self.e[0][z[N-1]]
        ML_amax=0
        for i in range(0,self.K):
            new_value=d[N-1,i]*self.e[i][z[N-1]]
            if new_value>ML:
                ML=new_value
                ML_amax=i

        #-------Backtracking to find maximizing path----
        #init:
        #x_ml[N-1]=np.argmax(d[N-1,:]*self.e[:,z[N-1]])
        x_ml[N-1]=ML_amax
        #recursion:
        for i in range(1,N):
            n=N-i-1
            x_ml[n]=T[n+1,x_ml[n+1]]

        return x_ml,ML