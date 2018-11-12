#------------------------------------------------------------------------
#Author: Sandro Giacomuzzi
#Part of my Master Thesis at ETHZ
#Written at: 1.11.2018
#------------------------------------------------------------------------

import numpy as np
from learnhmm.hmm import Hmm
from learnhmm.distributions import Gaussian_distribution

import time


eps=0 #protect zero divition

class Gaussian_emission(Hmm):

    def __init__(self,K,D,init_distr=None,trans_mat=None,means=None,covars=None,print_every=5):
        #means and covars must be a list of np-arrays
        #---preprocess input-----
        init_distr,trans_mat=self.check_input(K,init_distr,trans_mat,print_every)
        if means is None:
            means=[]
            for i in range(0,K):
                means.append(self.get_rp_vector(D)) #should at least be init with k-mean on some data 
        else:
            if len(means)!=K or means[0].shape[0]!= D:
                raise ValueError('The shape of the means must be a list of length K with arrays of shape D!')
        if covars is None:
            covars=[]
            for i in range(0,K):
                covar=np.random.rand(D,D)
                covars.append(np.dot(covar,covar.transpose())) #better to initialize at least with k means
        else:
            if len(covars)!=K or covars[0].shape[0]!=D or covars[0].shape[1]!=D:
               raise ValueError('The shape of the covars must be a list of length K containing arrays of shape (D,D)!')
            for i in range(0,K):
                if not self.check_symmetric_and_posdef(covars[i]):
                    raise ValueError('At least one of the covariance matrices is not symmetric and positive definite!')
        #------------------
        Hmm.__init__(self,'continuous',K,init_distr,trans_mat,print_every)#init from super class HMM
        #the emissions and its properties are subclass specific
        e=[]
        for i in range(0,K):
            e.append(Gaussian_distribution(means[i],covars[i]))
        self.e=e
        self.D=D

    def fit(self,z,n_iter=10,init='k_mean'):
        if init=='k_mean':
            [centroids,clusters,sum_distance]=self.k_mean(z,self.K,rep=3,n_iter=5)
            for k in range(0,self.K):
                self.e[k].mean=centroids[k]
                #calculate variance of clusters
                variances_k=np.var(clusters[k],axis=0)
                covar_k=np.diag(variances_k)
                self.e[k].covar=covar_k*100
        print('Start Baum-Welch...')
        return self.baumwelch(z,n_iter)

    def forward_scaled(self,z):
        #pi, initial distribution of hidden states
        #a, transition matrice of the MC
        #e, emission probability distributions for each state
        #z, observations (data)
        #This function returns the forward messages and the Likelihood

        N=z.shape[0] #size of sequence
        f_s=np.zeros((N,self.K)) #allocate memory for scaled forward messages
        #note that f_s[:,n] is a probability vector, namely, p(z_n|z_1,...,z_n-1)
        c=np.zeros(N) #scaling factors

        #initialize first message
        for k in range(0,self.K):
            f_s[0,k]=self.e[k].density(z[0])*self.pi[k]
            c[0]+=f_s[0,k]
        c[0]=np.sum(f_s[0,:])
        f_s[0,:]=f_s[0,:]/c[0]

        #forward propagation
        c_n_times_f_s=np.zeros(self.K)
        for n in range(1,N):
            for l in range(0,self.K):
                L_prev=0
                for k in range(0,self.K):
                    L_prev+=f_s[n-1,k]*self.a[k,l]
                c_n_times_f_s[l]=self.e[l].density(z[n])*L_prev
            c[n]=np.sum(c_n_times_f_s) #c_n is the normalizing coefficient
            f_s[n,:]=c_n_times_f_s/c[n]
            
            #---look for nans to debug
            if np.isnan(np.min(f_s[n,:])):
                print(c_n_times_f_s)
                print(c[n])
                for l in range(0,self.K):
                    L_prev=0
                    for k in range(0,self.K):
                        L_prev+=f_s[n-1,k]*self.a[k,l]
                    c_n_times_f_s[l]=self.e[l].density(z[n])*L_prev
                    print('L: '+str(L_prev))
                    print('e: '+str(self.e[l].density(z[n])))
                    print('z[n]: '+str(z[n]))
                    print(-(1/2)*np.einsum('i,i',z[n]-self.e[l].mean,np.einsum('ij,j', self.e[l].invcovar, z[n]-self.e[l].mean)))
                raise ValueError('nan error')

        LL=np.sum(np.log(c))
        return f_s,c,LL

    def backward_scaled(self,z,c):
        #a, transition matrice of the MC
        #e, emission probability distributions for each state
        #z, observations (data)
        #This function returns the scaled backward messages
        # c, the scaling factors computed in the forward phase
        z_shape=z.shape
        N=z.shape[0] #size of sequence
        b_s=np.zeros((N,self.K)) #allocate memory for scaled backward messages

        #initialize first message
        b_s[N-1,:]=1 

        #backward propagation
        for n in range(1,N):
            for l in range(0,self.K):
                for k in range(0,self.K):
                    #b_s[N-n-1,l]+=self.e[k,z_hot[N-n]]*b_s[N-n,k]*self.a[l,k]
                    b_s[N-n-1,l]+=self.e[k].density(z[N-n])*b_s[N-n,k]*self.a[l,k]
            b_s[N-n-1,:]/=(c[N-n]+eps)

        return b_s

    def get_gamma_ceta(self,z):
        #z must be a sequence (unwrapped from the list)
        N=z.shape[0] #sequence length
        [f_s,c,LL]=self.forward_scaled(z)
        b_s=self.backward_scaled(z,c)

        g=f_s*b_s
        ceta=np.zeros((N,self.K,self.K))
        for n in range(0,N-1):
            for k in range(0,self.K):
                for l in range(0,self.K):
                    ceta[n,l,k]=f_s[n,l]*self.a[l,k]*b_s[n+1,k]*self.e[k].density(z[n+1])/c[n+1]

        return g,ceta,LL
        #return g,ceta

    def m_step_emission(self,g_i,z):
        S=len(z)
        occurence_s=np.zeros(self.K)
        g_N_sum=np.zeros(self.K)

        for s in range(0,S):
            g_N_sum+=np.sum(g_i[s],axis=0) #the sum in the denominator

    #---update means--- 
        for k in range(0,self.K):
            enumerator_mean=0
            for s in range(0,S):
                for ns in range(0,g_i[s].shape[0]):
                    enumerator_mean+=g_i[s][ns,k]*z[s][ns,:]
            self.e[k].update_parameters(mean=enumerator_mean/g_N_sum[k])

    #---update covar with updated mean--- 
        for k in range(0,self.K):
            enumerator_covar=0
            for s in range(0,S):
                for ns in range(0,g_i[s].shape[0]):
                    derivation=z[s][ns,:]-self.e[k].mean
                    enumerator_covar+=g_i[s][ns,k]*np.outer(derivation,derivation)
            covar=enumerator_covar/g_N_sum[k]

            while not self.e[k].check_pos_def(covar):
                print('Warning: covariance matrix is not strictly positive definite! Reinitializing emission...')
                #initialize with average of the other emissions (for D=1 a collapse should not happen):                
                mean_new=self.e[k].mean
                covar_new=np.zeros((self.D,self.D))
                for i in range(0,self.K):
                    mean_new+=self.e[i].mean
                    covar_new+=self.e[i].covar
                for j in range (k+1,self.K):
                    mean_new+=self.e[j].mean
                    covar_new+=self.e[j].covar
                covar=covar_new/(self.K-1)
                print(covar)
                self.e[k].update_parameters(mean=mean_new/self.K)

            self.e[k].update_parameters(covar=covar)
            

                

    def sample_observation(self,z,L):
        #pi,a,e, the model parameters
        #z, an initial seed of observations
        #L, the number of predicted points
        #returns an oversvation sample

        N=z.shape[0] #size of sequence
        z_sampled=np.zeros((N+L,self.D))
        z_sampled[0:N,:]=z
        likelihood_of_means=np.zeros(self.K) #pred_distr[k]=p(z_N+1=k|z)
        pred_hidden=np.zeros(self.K) #pred_hidden[k]=p(z_N+1=k|x^N) 
        
        [f_s,c,Likelihood]=self.forward_scaled(z_sampled[0:N,:])
        f_s=f_s[N-1,:]
        for s in range(0,L):
            likelihood_of_means*=0
            pred_hidden*=0
            for k in range(0,self.K):
                for l in range(0,self.K):
                    pred_hidden[k]+=self.a[l,k]*f_s[l]
                #Heurestic ML-sampling (assuming)
                likelihood_of_means[k]=self.e[k].density(self.e[k].mean)*pred_hidden[k]
            k_max=np.argmax(likelihood_of_means)
            z_sampled[N+s,:]=self.e[np.argmax(likelihood_of_means)].mean
            [f_s,c]=self.scaled_forward_recursion_step(f_s,z_sampled[N+s,:])
        return z_sampled


    def scaled_forward_recursion_step(self,f_s_n,z_n):
        #f_s_n, the forward message at sequence point n
        #z_n,the n-th observation
        #K=f_s_n.shape[0]
        c_n_times_f_s=np.zeros(self.K)
        for l in range(0,self.K):
            L_prev=0
            for k in range(0,self.K):
                L_prev+=f_s_n[k]*self.a[k,l]
            #c_n_times_f_s[l]=self.e[l,np.argmax(z_n)]*L_prev
            c_n_times_f_s[l]=self.e[l].density(z_n)*L_prev
        c=np.sum(c_n_times_f_s)
        f_s_np1=c_n_times_f_s/c
        return f_s_np1,c 