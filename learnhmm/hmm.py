#------------------------------------------------------------------------
#Author: Sandro Giacomuzzi
#Part of my Master Thesis at ETHZ
#Written at: 1.11.2018
#------------------------------------------------------------------------

import numpy as np
from learnhmm import distributions
import time


eps=1e-8#protect zero division

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

    def k_mean(self,sequences,K,rep=1,n_iter=1000):
        #sequences is a list of sequences, which will then be concatenated
        epsilon=1e-3 #convergence criterion

        #-----concatenate sequences:
        data=sequences[0]
        for l in range(1,len(sequences)):
            data=np.concatenate((data,sequences[l]))
        N=data.shape[0]#amount of data
        D=data.shape[1]#dimension of a data point

        best_model=[]
        #-----run k-means rep times
        for rep in range(0,rep):
        
            #----init centroids:----
            centroids=np.zeros((K,D))#allocate memory
            mean=np.mean(data,axis=0)
            sigma=np.std(data,axis=0)
            #initialize centroids
            for d in range(0,D):
                centroids[:,d]=np.random.normal(mean[d],sigma[d],K)

            #-----k-mean iterations----
            print('start k-mean repetition: '+str(rep+1)+'...')
            clusters = [[] for _ in range(K)]
            #E-step: assign points to neaerest centroid
            last_distance=1e10
            new_distance=1e9
            best_model=[centroids,clusters,last_distance]
            iter=0
            while last_distance-new_distance>epsilon and iter<n_iter:
                last_distance=new_distance
                new_distance=0
                clusters = [[] for _ in range(K)]
                for n in range(0,N):
                    squared_distance=np.sum(np.power(data[n,:]-centroids[:,:],2),axis=1)#broadcasting
                    #squared_distance = numpy.linalg.norm(data[n,:]-centroids[:,:])
                    idx_min=np.argmin(squared_distance)
                    clusters[idx_min].append(data[n,:].tolist())
                    new_distance+=np.power(squared_distance[idx_min],1/2)

                #M-step: move centroids to cluster center
                for k in range(0,K):
                    if clusters[k]:
                        centroids[k,:]=np.mean(clusters[k],axis=0)
                iter+=1
                if np.mod(iter,5)==0:
                    print('iter: '+str(iter))
            if new_distance<best_model[2]:
                best_model=[centroids,clusters,new_distance]

        for k in range(0,K):
            best_model[1][k]=np.asarray(best_model[1][k])

        return best_model
       

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
            self.pi=g_1_sum/(g_partition+eps)

            #state tranistion matrix a
            ceta_s_sum=np.zeros((self.K,self.K))
            g_s_sum=np.zeros(self.K)+1e-3
            for s in range(0,S):
                ceta_sum=np.sum(ceta_i[s],axis=0)
                ceta_s_sum+=ceta_sum
                g_s_sum+=np.sum(ceta_sum,axis=1)
            #print('ceta_sum'+str(ceta_sum))

            #update transition matrix
            for k in range(0,self.K):
                #print('g_s_sum'+str(g_s_sum))
                self.a[:,k]=ceta_s_sum[:,k]/(g_s_sum[:]+eps)

            #update emission distribution e
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
                print('Epoch: '+str(iter)+', Training time: '+str(int(current_time))+'s, Likelihood: '+str(LL[iter+1]))
                print_time=current_time+self.print_every
            
            #----check consistency----
            # print('check g:' +str(1e-10>(N-1-np.sum(g))))
            # print('check ceta:' +str(1e-10>(N-1-np.sum(ceta))))        
            # print('check g_sum: ' +str((1e-10>np.sum(np.abs(g_sum-np.sum(g[0:-1,:],axis=0))))))


        print('Total training time: '+str(time.time()-start_time))
        return LL





    
