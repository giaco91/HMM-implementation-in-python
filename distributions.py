import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class Gaussian_distribution():

    def __init__(self,mean,covar):
        self.D=mean.shape[0]
        self.update_parameters(mean,covar)

    def check_symmetric(self,A):
        symmetric=np.allclose(A, A.T)
        return symmetric
    def check_pos_def(self,A):
        posdef=np.all(np.linalg.eigvals(A) > 0)
        return posdef


    def update_parameters(self,mean=None,covar=None):
        #---this check should be skipped for efficiency if used often and safely
        if mean is None:
            pass
        elif not mean.shape[0] == self.D:
            raise ValueError('The dimensions of the mean vector and the covariance matrix are not consistent.')
        else:
            self.mean=mean
        if covar is None:
            pass
        elif not ((covar.shape[0]==self.D and covar.shape[1]==self.D) and self.check_symmetric(covar)):
            raise ValueError('The covariance matrice is not symmetric!')
        elif not self.check_pos_def(covar):
            print('Warning: matrix is not strictly positive definite')
            print(covar)
            #try to make it positive definite
            ew=np.linalg.eigvals(covar)
            eps=-np.min(ew)+1e-1
            print('eps: '+str(eps))
            diags=np.ones(self.D)*eps
            covar+=np.diag(diags)
            posdef=np.all(np.linalg.eigvals(covar) > 0)
            if not posdef:
                print('covar still not strictly positive definite')
        else:
            self.covar=covar
            self.invcovar=np.linalg.inv(covar)
            self.density_proportionalfactor=np.power(2*np.pi,-self.D/2)*np.power(np.linalg.det(self.covar),-1/2)

    def density(self,x):
        if x.shape[0]!=self.D:
            raise ValueError('The evaluation point x has not dimension D!')
        return self.density_proportionalfactor*np.exp(-(1/2)*np.einsum('i,i',x-self.mean,np.einsum('ij,j', self.invcovar, x-self.mean)))



#----test code plot density of Gaussin in case of D=2-----

# covar1=np.array([[1,0],[0,1]])
# covar2=np.array([[1,0],[0,2]])
# mean1=np.array([0,0])
# mean2=np.array([2,0])

# Gauss_distr=Gaussian_distribution(mean2, covar2).density

# x = np.linspace(-5, 5, 30)
# y = np.linspace(-5, 5, 30)

# z=np.zeros((x.shape[0],y.shape[0]))
# for i in range(0,x.shape[0]):
#     for j in range(0,y.shape[0]):
#         z[i,j]=Gauss_distr(np.array([x[i],y[j]]))

# X, Y = np.meshgrid(x, y)
# Z=z
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z');
# plt.show()


