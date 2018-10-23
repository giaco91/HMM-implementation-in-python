#------------------------------------------------------------------------
#Author: Sandro Giacomuzzi
#Part of my Master Thesis at ETHZ
#Written at: 1.11.2018
#------------------------------------------------------------------------

import numpy as np
from distributions import *
import time


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


#super class
class Hmm():
    def __init__(self, emission_type,num_hiddenstates,init_prob,transition_matrix,print_every):
        self.emission_type=emission_type
        self.K=num_hiddenstates
        self.pi=init_prob
        self.a=transition_matrix
        self.print_every=print_every #prints information about the training states ever print_every seconds
    
    def sample_hiddenpath(self,L):
        x=np.zeros(L).astype(int)

        #ML sampling
        x[0]=np.argmax(self.pi)
        for l in range(1,L):
            x[l]=np.argmax(self.a[x[l-1],:])
        return x

    def check_symmetric_and_posdef(self,A):
        return np.allclose(A, A.T) and np.all(np.linalg.eigvals(A) > 0)

    def sample_from_hiddenpath(self,x):
        #x,a hidden path
        L=x.shape[0]
        z=np.zeros((L,self.D))
        #ML sampling
        for l in range(0,L):
            #z[l,np.argmax(self.e[x[l],:])]=1
            z[l,np.argmax(self.e[x[l]])]=1
        return z

    def get_rp_vector(self,N):
        #np.random.seed(2), Baum-Welch doesnt work with seed! WHY?
        #N, size of probability vector
        epsilon=0.1#to make sure that they are not too small
        P=np.random.rand(N)
        P+=epsilon
        P/=np.sum(P)
        return P

    
    def check_input(self,K,init_distr,trans_mat,print_every):
        if print_every<1:
            raise ValueError('print_every must be a positive number larger than one (seconds)!')
        if init_distr is None:
            init_distr=self.get_rp_vector(K)
        else:
            if init_distr.shape[0]!=K:
                raise ValueError('The shape of the initial probability vector must (K,)!')
            if not self.check_probability_vec(init_distr):
                raise ValueError('The given initial distribution is not a proper probability vector!')
        if trans_mat is None:
            trans_mat=np.zeros((K,K))
            for k in range(0,K):
                trans_mat[k,:]=self.get_rp_vector(K)
        else: 
            if trans_mat.shape[0]!=K or trans_mat.shape[1]!=K:
                raise ValueError('The shape of the transition matrix must be (K,K)!')
            if not self.check_trans_mat(trans_mat):
                raise ValueError('The given transition matrix is not a proper probability table!')
        return init_distr,trans_mat        

    def check_trans_mat(self,a):
        #this funktion checks a to be a allowed transition matrix
        #i.e. that it conserves the l1 norm and has non-negative entries
        for k in range(0,a.shape[0]):
            if not self.check_probability_vec(a[k,:]):
                return False
        return True

    def check_probability_vec(self,p):
        #checks if sum-to-one and non-negativity is given in the vector p
        check=True
        crit=1e-9
        if abs(1-np.sum(p))>crit:
            check=False
        for i in range(0,p.shape[0]):
            if p[i]<0:
                check=False
        return check

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
            c_n_times_f_s[l]=self.e[l][np.argmax(z_n)]*L_prev
        c=np.sum(c_n_times_f_s)
        f_s_np1=c_n_times_f_s/c
        return f_s_np1,c  

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
        z=self.one_hot_decoding(z)

        #initialize first message
        for k in range(0,self.K):
            #f_s[0,k]=self.e[k,z_hot[0]]*self.pi[k]
            f_s[0,k]=self.e[k][z[0]]*self.pi[k]
            c[0]+=f_s[0,k]
        f_s[0,:]=f_s[0,:]/c[0]

        #forward propagation
        c_n_times_f_s=np.zeros(self.K)
        for n in range(1,N):
            for l in range(0,self.K):
                L_prev=0
                for k in range(0,self.K):
                    L_prev+=f_s[n-1,k]*self.a[k,l]
                #c_n_times_f_s[l]=self.e[l,z_hot[n]]*L_prev
                c_n_times_f_s[l]=self.e[l][z[n]]*L_prev
            c[n]=np.sum(c_n_times_f_s) #c_n is the normalizing coefficient
            f_s[n,:]=c_n_times_f_s/c[n]

        L=np.prod(c)
        return f_s,c,L

    def backward_scaled(self,z,c):
        #a, transition matrice of the MC
        #e, emission probability distributions for each state
        #z, observations (data)
        #This function returns the scaled backward messages
        # c, the scaling factors computed in the forward phase
        z_shape=z.shape
        N=z.shape[0] #size of sequence
        b_s=np.zeros((N,self.K)) #allocate memory for scaled backward messages
        
        z=self.one_hot_decoding(z)

        #initialize first message
        b_s[N-1,:]=1 

        #backward propagation
        for n in range(1,N):
            for l in range(0,self.K):
                for k in range(0,self.K):
                    #b_s[N-n-1,l]+=self.e[k,z_hot[N-n]]*b_s[N-n,k]*self.a[l,k]
                    b_s[N-n-1,l]+=self.e[k][z[N-n]]*b_s[N-n,k]*self.a[l,k]
            b_s[N-n-1,:]/=c[N-n]

        return b_s

    def get_gamma_ceta(self,z):
        #z must be a sequence (unwrapped from the list)
        N=z.shape[0] #sequence length
        [f_s,c,L]=self.forward_scaled(z)
        LL=np.log(L)
        b_s=self.backward_scaled(z,c)

        z=self.one_hot_decoding(z)

        g=f_s*b_s
        ceta=np.zeros((N,self.K,self.K))
        for n in range(0,N-1):
            for k in range(0,self.K):
                for l in range(0,self.K):
                    #ceta[n,l,k]=f_s[n,l]*self.a[l,k]*b_s[n+1,k]*self.e[k,z_hot[n+1]]/c[n+1]
                    ceta[n,l,k]=f_s[n,l]*self.a[l,k]*b_s[n+1,k]*self.e[k][z[n+1]]/c[n+1]

        return g,ceta,LL

    def baumwelch(self,z,n_iter):
        #pi, initial distribution of hidden states
        #a, transition matrice of the MC
        #e, emission probability distributions for each state
        #z, observations (data)
        #This function returns the ML parameters for the EM procedure

        #notation:
        #g[n,l]=f[n,l]*b[n,l]/p(z^n)=p(x_n=l|z^n), responsibility
        #note: g[n,:] is a probabilidty vecator
        #ceta[n,l,k]=p(x_n=l,x_n+1 = k|z^n)
        #note: g[n,l]=np.sum(ceta,axis=2)

        start_time = time.time()
        print_time=self.print_every
        S=len(z) #amount of sequences
        LL=np.zeros(n_iter+1)#store the log-likelihood
        g_i=[]#store the responsibilities for all sequences
        ceta_i=[]#store the cetas for all sequences

        #-----first E-step------
        for s in range(0,S):
            [g,ceta,LL_s]=self.get_gamma_ceta(z[s])
            g_i.append(g)
            ceta_i.append(ceta)
            LL[0]+=LL_s #likelihood of sequnces are factors -> log-likelihood is sum
        #improvement=np.zeros((n_iter))#store (ln((L_i)-ln(L_i-1))/ln(L_i-1)

        for iter in range(0,n_iter):

            #----M-step for multiple sequences----
            
            #initial hidden state distr.
            g_1_sum=np.zeros(self.K)
            g_partition=0
            for s in range(0,S):
                g_1_sum+=g_i[s][0,:]
                g_partition+=np.sum(g_i[s][0,:])
            self.pi=g_1_sum/g_partition

            #state tranistion matrix a
            ceta_s_sum=np.zeros((self.K,self.K))
            g_s_sum=np.zeros(self.K)
            for s in range(0,S):
                ceta_sum=np.sum(ceta_i[s],axis=0)
                ceta_s_sum+=ceta_sum
                g_s_sum+=np.sum(ceta_sum,axis=1)
            for k in range(0,self.K):
                print('g_s_sum'+str(g_s_sum))
                self.a[:,k]=ceta_s_sum[:,k]/g_s_sum[:]

            #emission distribution e
            self.m_step_emission(g_i,z)
            # occurence_s=np.zeros(self.K)
            # g_N_sum=np.zeros(self.K)
            
            # for s in range(0,S):
            #     g_N_sum+=np.sum(g_i[s],axis=0)    
            # for d in range(0,self.D):
            #     occurence_s=0
            #     for s in range(0,S):
            #         occurence_s+=np.sum(g_i[s]*np.expand_dims(z[s][:,d], axis=1),axis=0)
            #     #self.e[:,d]=occurence_s/g_N_sum
            #     helper=occurence_s/g_N_sum
            #     for i in range(0,self.K):
            #         self.e[i][d]=helper[i]

            #---E-step-----        
            for s in range(0,S):
                [g,ceta,LL_s]=self.get_gamma_ceta(z[s])
                g_i[s]=g
                ceta_i[s]=ceta
                LL[iter+1]+=LL_s #likelihood of sequnces are factors -> log-likelihood is sum

            current_time=time.time()-start_time
            if current_time>print_time:
                print('Epoch: '+str(iter)+', Training time: '+str(int(current_time))+'s, Likelihood: '+str(LL_s))
                print_time=current_time+self.print_every
            #----check consistency----
            # print('check g:' +str(1e-10>(N-1-np.sum(g))))
            # print('check ceta:' +str(1e-10>(N-1-np.sum(ceta))))        
            # print('check g_sum: ' +str((1e-10>np.sum(np.abs(g_sum-np.sum(g[0:-1,:],axis=0))))))


        print('Total training time: '+str(time.time()-start_time))
        return LL

#The actual hmm-graphs are child classes.
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
                covars.append(np.dot(covar,covar.transpose())) #better to initialize with scaling order of some data, e.g. GMM
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

    def fit(self,z,n_iter):
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
        f_s[0,:]=f_s[0,:]/c[0]

        #forward propagation
        c_n_times_f_s=np.zeros(self.K)
        for n in range(1,N):
            for l in range(0,self.K):
                L_prev=0
                for k in range(0,self.K):
                    L_prev+=f_s[n-1,k]*self.a[k,l]
                #c_n_times_f_s[l]=self.e[l,z_hot[n]]*L_prev
                c_n_times_f_s[l]=self.e[l].density(z[n])*L_prev
            c[n]=np.sum(c_n_times_f_s) #c_n is the normalizing coefficient
            f_s[n,:]=c_n_times_f_s/c[n]

        L=np.prod(c)
        return f_s,c,L

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
            b_s[N-n-1,:]/=c[N-n]

        return b_s

    def get_gamma_ceta(self,z):
        #z must be a sequence (unwrapped from the list)
        N=z.shape[0] #sequence length
        [f_s,c,L]=self.forward_scaled(z)
        LL=np.log(L)
        b_s=self.backward_scaled(z,c)

        g=f_s*b_s
        ceta=np.zeros((N,self.K,self.K))
        for n in range(0,N-1):
            for k in range(0,self.K):
                for l in range(0,self.K):
                    ceta[n,l,k]=f_s[n,l]*self.a[l,k]*b_s[n+1,k]*self.e[k].density(z[n+1])/c[n+1]

        return g,ceta,LL

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
            self.e[k].update_parameters(covar=enumerator_covar/g_N_sum[k])

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

    
