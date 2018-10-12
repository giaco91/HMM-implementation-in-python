#------------------------------------------------------------------------
#Author: Sandro Giacomuzzi
#Part of my Master Thesis at ETHZ
#Written at: 1.11.2018
#------------------------------------------------------------------------

import numpy as np
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
    def __init__(self, emission_type,num_hiddenstates,init_prob,transition_matrix):
        self.emission_type=emission_type
        self.K=num_hiddenstates
        self.pi=init_prob
        self.a=transition_matrix
    
    def sample_hiddenpath(self,L):
        pi=self.pi
        a=self.a
        x=np.zeros(L).astype(int)
        
        #ML sampling
        x[0]=np.argmax(pi)
        for l in range(1,L):
            x[l]=np.argmax(a[x[l-1],:])
        return x

    def sample_from_hiddenpath(self,x):
        #x,a hidden path
        L=x.shape[0]
        z=np.zeros((L,self.D))
        #ML sampling
        for l in range(0,L):
            z[l,np.argmax(self.e[x[l],:])]=1
        return z

    def get_rp_vector(self,N):
        #np.random.seed(2), Baum-Welch doesnt work with seed! WHY?
        #N, size of probability vector
        epsilon=0.1#to make sure that they are not too small
        P=np.random.rand(N)
        P+=epsilon
        P/=np.sum(P)
        return P

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
        K=self.K
        a=self.a
        e=self.e
        c_n_times_f_s=np.zeros(K)
        for l in range(0,K):
            L_prev=0
            for k in range(0,K):
                L_prev+=f_s_n[k]*a[k,l]
            c_n_times_f_s[l]=e[l,np.argmax(z_n)]*L_prev
        c=np.sum(c_n_times_f_s)
        f_s_np1=c_n_times_f_s/c
        return f_s_np1,c  

    def forward_scaled(self,z):
        #pi, initial distribution of hidden states
        #a, transition matrice of the MC
        #e, emission probability distributions for each state
        #z, observations (data)
        #This function returns the forward messages and the Likelihood
        
        K=self.K #number of hidden states
        pi=self.pi
        e=self.e
        a=self.a
        D=self.D #number of output states (for discrete ouput space)
        z_shape=z.shape
        N=z_shape[0] #size of sequence
        f_s=np.zeros((N,K)) #allocate memory for scaled forward messages
        #note that f_s[:,n] is a probability vector, namely, p(z_n|z_1,...,z_n-1)
        c=np.zeros(N) #scaling factors
        z_hot=self.one_hot_decoding(z)

        #initialize first message
        for k in range(0,K):
            f_s[0,k]=e[k,z_hot[0]]*pi[k]
            c[0]+=f_s[0,k]
        f_s[0,:]=f_s[0,:]/c[0]

        #forward propagation
        c_n_times_f_s=np.zeros(K)
        for n in range(1,N):
            for l in range(0,K):
                L_prev=0
                for k in range(0,K):
                    L_prev+=f_s[n-1,k]*a[k,l]
                c_n_times_f_s[l]=e[l,z_hot[n]]*L_prev
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

        K=self.K #number of hidden states
        D=self.D
        a=self.a
        e=self.e
        z_shape=z.shape
        N=z_shape[0] #size of sequence
        b_s=np.zeros((N,K)) #allocate memory for scaled backward messages
        
        z_hot=self.one_hot_decoding(z)

        #initialize first message
        b_s[N-1,:]=1 

        #backward propagation
        for n in range(1,N):
            for l in range(0,K):
                for k in range(0,K):
                    b_s[N-n-1,l]+=e[k,z_hot[N-n]]*b_s[N-n,k]*a[l,k]
            b_s[N-n-1,:]/=c[N-n]

        return b_s

#The actual hmm-graphs are child classes.
class Discrete_emission(Hmm):

    def __init__(self,K,D,init_distr=None,trans_mat=None,emission_mat=0):
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
        if emission_mat is None:
            emission_mat=np.zeros((K,D))
            for k in range(0,K):
                emission_mat[k,:]=self.get_rp_vector(D) 
        else:
            if emission_mat.shape[0]!=K or emission_mat.shape[1]!=D:
                raise ValueError('The shape of the transition matrix must be (K,D)!')
            if not self.check_emission_mat(emission_mat):
                raise ValueError('The given emission matrix is not a proper probability table!')
        Hmm.__init__(self,'discrete_emission',K,init_distr,trans_mat)
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


    def check_emission_mat(self,e):
        #this funktion checks a to be a allowed emission
        #i.e. that it is a valid probability dirstibution for each k
        for k in range(0,e.shape[0]):
            if not self.check_probability_vec(e[k,:]):
                return False
        return True


    def sample_observation(self,z,L):
        #pi,a,e, the model parameters
        #z, an initial seed of observations
        #L, the number of predicted points
        #returns an oversvation sample

        K=self.K #number of hidden states
        D=self.D #number of output states (for discrete ouput space)
        pi=self.pi
        a=self.a
        e=self.e
        z_shape=z.shape
        N=z_shape[0] #size of sequence
        z_sampled=np.zeros((N+L,D))
        z_sampled[0:N,:]=z
        pred_distr=np.zeros(D) #pred_distr[k]=p(z_N+1=k|z)
        pred_hidden=np.zeros(K) #pred_hidden[k]=p(z_N+1=k|x^N) 
        
        [f_s,c,Likelihood]=self.forward_scaled(z_sampled[0:N,:])
        f_s=f_s[N-1,:]
        for s in range(0,L):
            pred_distr*=0
            pred_hidden*=0
            for k in range(0,K):
                for l in range(0,K):
                    pred_hidden[k]+=a[l,k]*f_s[l]
                for d in range(0,D):
                    pred_distr[d]+=e[k,d]*pred_hidden[k]
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

        K=self.K #number of hidden states
        D=self.D
        pi=self.pi
        a=self.a
        e=self.e
        z_shape=z.shape
        N=z_shape[0] #size of sequence

        z=self.one_hot_decoding(z)

        #Denote: 
        #d[t,l]=max_x^{t-1}p(x^{t-1},x_t=l,z^n)
        #T[n,l]=argmax_k{d[n-1,k]*a[k,l]}, the tracker
        #x_ml = argmax_x^n{p(x^n,z^n)}

        d=np.zeros((N,K))
        T=np.zeros((N,K)).astype(int)#note: T[0,:] stays unused, but I still allocate it for convenience
        x_ml=np.zeros((N)).astype(int)

        #init
        for k in range(0,K):
            d[0,k]=pi[k]*e[k,z[0]]

        #recursion
        for n in range(1,N):
            for l in range(0,K):
                d[n,l]=np.amax(d[n-1,:]*a[:,l])*e[l,z[n]]
                T[n,l]=np.argmax(d[n-1,:]*a[:,l])
        #ML
        ML=np.amax(d[N-1,:]*e[:,z[N-1]])

        #Backtracking to find maximizing path
        #init:
        x_ml[N-1]=np.argmax(d[N-1,:]*e[:,z[N-1]])
        #recursion:
        for i in range(1,N):
            n=N-i-1
            x_ml[n]=T[n+1,x_ml[n+1]]

        return x_ml,ML

    # def forward(pi,a,e,z):
    #     #pi, initial distribution of hidden states
    #     #a, transition matrice of the MC
    #     #e, emission probability distributions for each state
    #     #z, observations (data)
    #     #This function returns the forward messages and the Likelihood
        
    #     K=pi.shape[0] #number of hidden states
    #     z_shape=z.shape
    #     N=z_shape[0] #size of sequence
    #     D=z_shape[1] #number of output states (for discrete ouput space)
    #     f=np.zeros((N,K)) #allocate memory for forward messages
        
    #     z_hot=one_hot_decoding(z)

    #     #initialize first message
    #     for k in range(0,K):
    #         f[0,k]=e[k,z_hot[0]]*pi[k]

    #     #forwarad propagation
    #     for n in range(1,N):
    #         for l in range(0,K):
    #             L_prev=0
    #             for k in range(0,K):
    #                 L_prev+=f[n-1,k]*a[k,l]
    #             f[n,l]=e[l,z_hot[n]]*L_prev
        
    #     L=np.sum(f[N-1,:])
    #     return f,L


    # def backward(a,e,z):
    #     #a, transition matrice of the MC
    #     #e, emission probability distributions for each state
    #     #z, observations (data)
    #     #This function returns the backward messages

    #     K=a.shape[0] #number of hidden states
    #     z_shape=z.shape
    #     N=z_shape[0] #size of sequence
    #     D=z_shape[1] #number of output states (for discrete ouput space)
    #     b=np.zeros((N,K)) #allocate memory for backward messages
        
    #     z_hot=one_hot_decoding(z)

    #     #initialize first message
    #     b[N-1,:]=1

    #     #backward propagation
    #     for n in range(1,N):
    #         for l in range(0,K):
    #             for k in range(0,K):
    #                 b[N-n-1,l]+=e[k,z_hot[N-n]]*b[N-n,k]*a[l,k]
    #     return b
     

    # def unscale_messages(f_s,b_s,c):
    #     #f_s, scaled forward messages
    #     #b_s, scaled backward messages
    #     #c, scaling factors

    #     f_s_shape=f_s.shape
    #     N=f_s_shape[0]
    #     K=f_s_shape[1]

    #     f_unscaled=np.zeros((N,K))
    #     L_n=1
    #     for i in range(0,N):
    #         L_n*=c[i]
    #         f_unscaled[i,:]=f_s[i,:]*L_n

    #     b_unscaled=np.zeros((N,K))
    #     b_unscaled[N-1,:]=1 #broadcasting
    #     L_n=1
    #     for i in range(1,N):
    #         n=N-i-1
    #         L_n*=c[n+1]
    #         b_unscaled[n,:]=b_s[n,:]*L_n

    #     return f_unscaled, b_unscaled


    # def check_forward_backward_consistency(pi,a,e,z):

    #     K=a.shape[0] #number of hidden states
    #     z_shape=z.shape
    #     N=z_shape[0] #size of sequence
    #     D=z_shape[1] #number of output states (for discrete ouput space)

    #     [f_s,c,L_s]=forward_scaled(pi,a,e,z)
    #     b_s=backward_scaled(a,e,z,c)
    #     [f,L]=forward(pi,a,e,z)
    #     b=backward(a,e,z)
        
    #     [f_unscaled, b_unscaled]=unscale_messages(f_s,b_s,c)

    #     eps=1e-10
    #     check_L=abs((L-np.prod(c)))<eps
    #     check_f_s=np.sum(np.abs(f-f_unscaled))<eps
    #     check_b_s=np.sum(np.abs(b-b_unscaled))<eps

    #     # print('is b_s*f_s the problem?: '+str(not(eps>np.sum(np.abs(L-np.sum(f_s[3,:]*b_s[3,:]))))))
        
    #     # print('check L and c')
    #     # print(check_L)    
        
    #     # print('check f_s: ')
    #     # print(check_f_s)
        
    #     # print('check b_s: ')
    #     # print(check_b_s)

    #     return (check_L & check_f_s & check_b_s)


    # def get_gamma_ceta(pi,a,e,z):
    #     #z must be a sequence (unwrapped from the list)
    #     K=a.shape[0] #number of hidden states
    #     z_shape=z.shape
    #     N=z_shape[0] #sequence length
    #     D=z_shape[1] #number of output states (for discrete ouput space)

    #     z_hot=one_hot_decoding(z)
    #     [f_s,c,L]=forward_scaled(pi,a,e,z)
    #     LL=np.log(L)
    #     b_s=backward_scaled(a,e,z,c)

    #     g=f_s*b_s
    #     ceta=np.zeros((N,K,K))
    #     for n in range(0,N-1):
    #         for k in range(0,K):
    #             for l in range(0,K):
    #                 ceta[n,l,k]=f_s[n,l]*a[l,k]*b_s[n+1,k]*e[k,z_hot[n+1]]/c[n+1]

    #     return g,ceta,LL

    # def baumwelch(pi,a,e,z,n_iter):
    #     #pi, initial distribution of hidden states
    #     #a, transition matrice of the MC
    #     #e, emission probability distributions for each state
    #     #z, observations (data)
    #     #This function returns the ML parameters for the EM procedure

    #     #notation:
    #     #g[n,l]=f[n,l]*b[n,l]/p(z^n)=p(x_n=l|z^n), responsibility
    #     #note: g[n,:] is a probabilidty vecator
    #     #ceta[n,l,k]=p(x_n=l,x_n+1 = k|z^n)
    #     #note: g[n,l]=np.sum(ceta,axis=2)

    #     start_time = time.time()

    #     K=a.shape[0] #number of hidden states
    #     D=z[0].shape[1] #number of output states (for discrete ouput space)
    #     S=len(z) #amount of sequences
    #     LL=np.zeros(n_iter+1)#store the log-likelihood
    #     g_i=[]#store the responsibilities for all sequences
    #     ceta_i=[]#store the cetas for all sequences

    #     #-----first E-step------
    #     for s in range(0,S):
    #         [g,ceta,LL_s]=get_gamma_ceta(pi,a,e,z[s])
    #         g_i.append(g)
    #         ceta_i.append(ceta)
    #         LL[0]+=LL_s #likelihood of sequnces are factors -> log-likelihood is sum
    #     #improvement=np.zeros((n_iter))#store (ln((L_i)-ln(L_i-1))/ln(L_i-1)

    #     for iter in range(0,n_iter):

    #         #----M-step for multiple sequences----
            
    #         #initial hidden state distr.
    #         g_1_sum=np.zeros(K)
    #         g_partition=0
    #         for s in range(0,S):
    #             g_1_sum+=g_i[s][0,:]
    #             g_partition+=np.sum(g_i[s][0,:])
    #         pi[:]=g_1_sum/g_partition

    #         #state tranistion matrix a
    #         ceta_s_sum=np.zeros((K,K))
    #         g_s_sum=np.zeros(K)
    #         for s in range(0,S):
    #             ceta_sum=np.sum(ceta_i[s],axis=0)
    #             ceta_s_sum+=ceta_sum
    #             g_s_sum+=np.sum(ceta_sum,axis=1)
    #         for k in range(0,K):
    #             a[:,k]=ceta_s_sum[:,k]/g_s_sum[:]

    #         #emission distribution e
    #         occurence_s=np.zeros(K)
    #         g_N_sum=np.zeros(K)
            
    #         for s in range(0,S):
    #             g_N_sum+=np.sum(g_i[s],axis=0)    
    #         for d in range(0,D):
    #             occurence_s=0
    #             for s in range(0,S):
    #                 occurence_s+=np.sum(g_i[s]*np.expand_dims(z[s][:,d], axis=1),axis=0)
    #                 print(occurence_s)
    #                 print(g_N_sum)
    #             e[:,d]=occurence_s/g_N_sum

    #         #---E-step multiple sequences-----        
    #         for s in range(0,S):
    #             [g,ceta,LL_s]=get_gamma_ceta(pi,a,e,z[s])
    #             g_i[s][:,:]=g
    #             ceta_i[s][:,:,:]=ceta
    #             LL[iter]+=LL_s #likelihood of sequnces are factors -> log-likelihood is sum




    #         #----check consistency----
    #         # print('check g:' +str(1e-10>(N-1-np.sum(g))))
    #         # print('check ceta:' +str(1e-10>(N-1-np.sum(ceta))))        
    #         # print('check g_sum: ' +str((1e-10>np.sum(np.abs(g_sum-np.sum(g[0:-1,:],axis=0))))))




    #     print('Baumwelch time: '+str(time.time()-start_time))
    #     return pi,a,e,LL