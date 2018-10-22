import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class Gaussian_distribution():

    def __init__(self,mean,covar):
        self.D=mean.shape[0]
        self.update_parameters(mean,covar)

    def check_symmetric_and_posdef(self,A):
        return np.allclose(A, A.T) and np.all(np.linalg.eigvals(A) > 0)

    def update_parameters(self,mean,covar):
        #---this check should be skipped for efficiency if used often and safely 
        if not ((mean.shape[0] == covar.shape[0]) and (mean.shape[0] == covar.shape[0])):
            raise ValueError('The dimensions of the mean vector and the covariance matrix are not consistent.')
        if not self.check_symmetric_and_posdef(covar):
            raise ValueError('The covariance matrice is not symmetric and positive definite!')
        self.mean=mean
        self.covar=covar
        self.invcovar=np.linalg.inv(covar)
        self.density_proportionalfactor=np.power(2*np.pi,-self.D/2)*np.power(np.linalg.det(covar),-1/2)

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


