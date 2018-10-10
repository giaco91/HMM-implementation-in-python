import numpy as np
import time

#-----------notation-----------
#K, number of hidden states
#N, sequence size
#D, size of ouput space
#pi, initial distribution of hidden states
#a, transition matrice of the MC
#e, emission probability distributions for each state
#x, hidden the states
#z, observations (data)
#f[n,k]=p(z_1,...,z_n,x_n=k)
#b[n,k]=p(z_n+1,...,z_N|x_n=k)

#------------------------------

def sample_hiddenpath(pi,a,L):
    x=np.zeros(L).astype(int)
    
    #ML sampling
    x[0]=np.argmax(pi)
    for l in range(1,L):
        x[l]=np.argmax(a[x[l-1],:])
    return x

def sample_from_hiddenpath(x,e):
    z=np.zeros((x.shape[0],e.shape[1]))
    L=x.shape[0]

    #ML sampling
    for l in range(0,L):
        z[l,np.argmax(e[x[l],:])]=1
    return z


def sample_observation(pi,a,e,z,L):
    #pi,a,e, the model parameters
    #z, an initial seed of observations
    #L, the number of predicted points
    #returns an oversvation sample

    K=pi.shape[0] #number of hidden states
    z_shape=z.shape
    N=z_shape[0] #size of sequence
    D=z_shape[1] #number of output states (for discrete ouput space)
    z_sampled=np.zeros((N+L,D))
    z_sampled[0:N,:]=z
    pred_distr=np.zeros(D) #pred_distr[k]=p(z_N+1=k|z)
    pred_hidden=np.zeros(K) #pred_hidden[k]=p(z_N+1=k|x^N) 
    
    [f_s,c,Likelihood]=forward_scaled(pi,a,e,z_sampled[0:N,:])
    f_s=f_s[N-1,:]
    for s in range(0,L):
        pred_distr*=0
        pred_hidden*=0
        #[f_s,c,L]=forward_scaled(pi,a,e,z_sampled[0:N+s,:])
        for k in range(0,K):
            for l in range(0,K):
                pred_hidden[k]+=a[l,k]*f_s[l]
                #pred_hidden[k]+=a[l,k]*f_s[N-1+s,l]
            for d in range(0,D):
                pred_distr[d]+=e[k,d]*pred_hidden[k]
        z_sampled[N+s,np.argmax(pred_distr)]=1 #ML-sampling
        [f_s,c]=scaled_forward_recursion_step(f_s,z_sampled[N+s,:],a,e)
    return z_sampled


def get_rp_vector(N):
    #np.random.seed(2), Baum-Welch doesnt work with seed! WHY?
    #N, size of probability vector
    e=0.1#to make sure that they are not too small
    P=np.random.rand(N)
    P+=e
    P/=np.sum(P)
    return P

def one_hot_decoding(z):
    #shape of z: (n,o)
    #assignes to each one-hot-vector an integer value
    z_shape=z.shape
    n=z.shape[0]
    o=z.shape[1]
    z_decoded=np.zeros(n)
    for i in range(0,n):
        k=0
        for j in range(0,o):
            if z[i,j]==1:
                k=j
        z_decoded[i]=k
    return z_decoded.astype(int)

def init(K,D):
    #Parameter initialization
    #N,sequence size
    #K,hidden states
    #D,output states
    
    pi=get_rp_vector(K)
    a=np.zeros((K,K))
    e=np.zeros((K,D))
    for k in range(0,K):
        a[k,:]=get_rp_vector(K)
        e[k,:]=get_rp_vector(D)

    sequences=[]
    #cyclic sequence that can be learned perfectly for K>=3
    z=np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0]]) #shape: (N,D) (1-hot encoding)
    sequences.append(z)

    return pi,a,e,sequences

def check_transmat(a):
    #this funktion checks a to be a allowed transition matrix
    #i.e. that it conserves the l1 norm
    crit=1e-9
    K=a.shape[0]
    check=True
    for k in range(0,K):
        s=0
        for l in range(0,K):
            s+=a[k,l]
        if abs(1-s)>crit:
            check=False
    return check

def check_emission(e):
    #this funktion checks a to be a allowed emission
    #i.e. that it is a valid probability dirstibution for each k
    crit=1e-9
    K=e.shape[0]
    check=True
    for k in range(0,K):
        if abs(1-np.sum(e[k,:]))>crit:
            check=False
    return check

def check_pi(pi):
    crit=1e-9
    check=True
    if abs(1-np.sum(pi))>crit:
        check=False
    return check

def scaled_forward_recursion_step(f_s_n,z_n,a,e):
    #f_s_n, the forward message at point n
    #the n-th observation
    K=f_s_n.shape[0]
    c_n_times_f_s=np.zeros(K)
    for l in range(0,K):
        L_prev=0
        for k in range(0,K):
            L_prev+=f_s_n[k]*a[k,l]
        c_n_times_f_s[l]=e[l,np.argmax(z_n)]*L_prev
    c=np.sum(c_n_times_f_s)
    f_s_np1=c_n_times_f_s/c
    return f_s_np1,c  

def forward_scaled(pi,a,e,z):
    #pi, initial distribution of hidden states
    #a, transition matrice of the MC
    #e, emission probability distributions for each state
    #z, observations (data)
    #This function returns the forward messages and the Likelihood
    
    K=pi.shape[0] #number of hidden states
    z_shape=z.shape
    N=z_shape[0] #size of sequence
    D=z_shape[1] #number of output states (for discrete ouput space)
    f_s=np.zeros((N,K)) #allocate memory for scaled forward messages
    #note that f_s[:,n] is a probability vector, namely, p(z_n|z_1,...,z_n-1)
    c=np.zeros(N) #scaling factors
    z_hot=one_hot_decoding(z)

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
        #f_s[n,:]=scaled_forward_recursion_step(f_s[n-1,:],z[n],a,e)

    L=np.prod(c)
    return f_s,c,L

def forward(pi,a,e,z):
    #pi, initial distribution of hidden states
    #a, transition matrice of the MC
    #e, emission probability distributions for each state
    #z, observations (data)
    #This function returns the forward messages and the Likelihood
    
    K=pi.shape[0] #number of hidden states
    z_shape=z.shape
    N=z_shape[0] #size of sequence
    D=z_shape[1] #number of output states (for discrete ouput space)
    f=np.zeros((N,K)) #allocate memory for forward messages
    
    z_hot=one_hot_decoding(z)

    #initialize first message
    for k in range(0,K):
        f[0,k]=e[k,z_hot[0]]*pi[k]

    #forwarad propagation
    for n in range(1,N):
        for l in range(0,K):
            L_prev=0
            for k in range(0,K):
                L_prev+=f[n-1,k]*a[k,l]
            f[n,l]=e[l,z_hot[n]]*L_prev
    
    L=np.sum(f[N-1,:])
    return f,L


def backward_scaled(a,e,z,c):
    #a, transition matrice of the MC
    #e, emission probability distributions for each state
    #z, observations (data)
    #This function returns the scaled backward messages
    # c, the scaling factors computed in the forward phase

    K=a.shape[0] #number of hidden states
    z_shape=z.shape
    N=z_shape[0] #size of sequence
    D=z_shape[1] #number of output states (for discrete ouput space)
    b_s=np.zeros((N,K)) #allocate memory for scaled backward messages
    
    z_hot=one_hot_decoding(z)

    #initialize first message
    b_s[N-1,:]=1 

    #backward propagation
    for n in range(1,N):
        for l in range(0,K):
            for k in range(0,K):
                b_s[N-n-1,l]+=e[k,z_hot[N-n]]*b_s[N-n,k]*a[l,k]
        b_s[N-n-1,:]/=c[N-n]

    return b_s

def backward(a,e,z):
    #a, transition matrice of the MC
    #e, emission probability distributions for each state
    #z, observations (data)
    #This function returns the backward messages

    K=a.shape[0] #number of hidden states
    z_shape=z.shape
    N=z_shape[0] #size of sequence
    D=z_shape[1] #number of output states (for discrete ouput space)
    b=np.zeros((N,K)) #allocate memory for backward messages
    
    z_hot=one_hot_decoding(z)

    #initialize first message
    b[N-1,:]=1

    #backward propagation
    for n in range(1,N):
        for l in range(0,K):
            for k in range(0,K):
                b[N-n-1,l]+=e[k,z_hot[N-n]]*b[N-n,k]*a[l,k]
    return b
 

def unscale_messages(f_s,b_s,c):
    #f_s, scaled forward messages
    #b_s, scaled backward messages
    #c, scaling factors

    f_s_shape=f_s.shape
    N=f_s_shape[0]
    K=f_s_shape[1]

    f_unscaled=np.zeros((N,K))
    L_n=1
    for i in range(0,N):
        L_n*=c[i]
        f_unscaled[i,:]=f_s[i,:]*L_n

    b_unscaled=np.zeros((N,K))
    b_unscaled[N-1,:]=1 #broadcasting
    L_n=1
    for i in range(1,N):
        n=N-i-1
        L_n*=c[n+1]
        b_unscaled[n,:]=b_s[n,:]*L_n

    return f_unscaled, b_unscaled


def check_forward_backward_consistency(pi,a,e,z):

    K=a.shape[0] #number of hidden states
    z_shape=z.shape
    N=z_shape[0] #size of sequence
    D=z_shape[1] #number of output states (for discrete ouput space)

    [f_s,c,L_s]=forward_scaled(pi,a,e,z)
    b_s=backward_scaled(a,e,z,c)
    [f,L]=forward(pi,a,e,z)
    b=backward(a,e,z)
    
    [f_unscaled, b_unscaled]=unscale_messages(f_s,b_s,c)

    eps=1e-10
    check_L=abs((L-np.prod(c)))<eps
    check_f_s=np.sum(np.abs(f-f_unscaled))<eps
    check_b_s=np.sum(np.abs(b-b_unscaled))<eps

    # print('is b_s*f_s the problem?: '+str(not(eps>np.sum(np.abs(L-np.sum(f_s[3,:]*b_s[3,:]))))))
    
    # print('check L and c')
    # print(check_L)    
    
    # print('check f_s: ')
    # print(check_f_s)
    
    # print('check b_s: ')
    # print(check_b_s)

    return (check_L & check_f_s & check_b_s)

def viterbi(pi,a,e,z):
    #pi, initial distribution of hidden states
    #a, transition matrice of the MC
    #e, emission probability distributions for each state
    #z, observations (data)
    #This function returns the ML hidden path.
    #Furthermore it returns the ML joint p(x^n,z^n). 

    K=a.shape[0] #number of hidden states
    z_shape=z.shape
    N=z_shape[0] #size of sequence
    D=z_shape[1] #number of output states (for discrete ouput space)

    z=one_hot_decoding(z)

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

def baumwelch(pi,a,e,z,n_iter):
    z=z[0]
    #pi, initial distribution of hidden states
    #a, transition matrice of the MC
    #e, emission probability distributions for each state
    #z, observations (data)
    #This function returns the ML parameters for the EM procedure

    #notation:
    #g[n,l]=f[n,l]*b[n,l]/p(z^n)=p(x_n=l|z^n)
    #note: g[n,:] is a probabilidty vecator
    #ceta[n,l,k]=p(x_n=l,x_n+1 = k|z^n)
    #note: g[n,l]=np.sum(ceta,axis=2)

    start_time = time.time()

    K=a.shape[0] #number of hidden states
    z_shape=z.shape
    N=z_shape[0] #size of sequence
    D=z_shape[1] #number of output states (for discrete ouput space)

    z_hot=one_hot_decoding(z)
    improvement=np.zeros((n_iter))#store (ln((L_i)-ln(L_i-1))/ln(L_i-1)
    LL=np.zeros(n_iter+1)
    for iter in range(0,n_iter):

        #----E-step---

        [f_s,c,L]=forward_scaled(pi,a,e,z)
        LL[iter]=np.log(L)
        b_s=backward_scaled(a,e,z,c)

        g=f_s*b_s
        ceta=np.zeros((N,K,K))
        for n in range(0,N-1):
            for k in range(0,K):
                for l in range(0,K):
                    ceta[n,l,k]=f_s[n,l]*a[l,k]*b_s[n+1,k]*e[k,z_hot[n+1]]/c[n+1]


        #---M-step---
        #initial hidden state dirstribution pi
        pi[:]=g[0,:]/np.sum(g[0,:])

        #state tranistion matrix a
        ceta_sum=np.sum(ceta,axis=0)
        g_sum=np.sum(ceta_sum,axis=1)#=np.sum(g[0:-1,:],axis=0)

        
        for k in range(0,K):
            a[:,k]=ceta_sum[:,k]/g_sum[:]


        #emission distribution e
        for d in range(0,D):
            e[:,d]=np.sum(g*np.expand_dims(z[:,d], axis=1),axis=0)/np.sum(g,axis=0)

        #----check consistency----
        # print('check g:' +str(1e-10>(N-1-np.sum(g))))
        # print('check ceta:' +str(1e-10>(N-1-np.sum(ceta))))        
        # print('check g_sum: ' +str((1e-10>np.sum(np.abs(g_sum-np.sum(g[0:-1,:],axis=0))))))


    [f_s,c,L]=forward_scaled(pi,a,e,z)
    LL[n_iter]=np.log(L)
    print('Baumwelch time: '+str(time.time()-start_time))
    return pi,a,e,LL
