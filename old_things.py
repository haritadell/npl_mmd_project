#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:42:27 2021

@author: HaritaDellaporta
"""

#%%
# Check gradient 
m = 1000              # number of simulated samples
n = 1500              # number of true samples
theta = np.array([3,1,1,-np.log(2)])     # true theta
d = 1
p = 4                 # dimensions of data
#p = d                 # dimensions of parameter space
#s = 2                 # standard deviation of the model
l = -1                # lengthscale (l=-1 for median heuristic)

y, z = sample_gandk_outl(m,d,theta,n_cont=0)
x, _ = sample_gandk_outl(n,d,theta,n_cont=0)

kxx = k(x,x,l)
kxy = k(y,x,l)
kyy = k(y,y,l)
k1yy = kyy[1]
k1xy = kxy[1]
k21xx = kxx[2]
model = models.g_and_k_model(m,d)
grad_g = model.grad_generator(z, theta)

# check gradient using finite differences
weights = (1/n)*np.ones(n)
# check gradient using finite differences
par = 2
theta_check = deepcopy(theta)
theta_check[par] = theta_check[par] + 0.00000001
y_check = gen_gandk(z,theta_check)
print('check gradient of the MMD^2 approximation:')
print((MMD_approx(n,m,weights,kxx[0],k(y_check,x,l)[0],k(y_check,y_check,l)[0])-MMD_approx(n,m,weights,kxx[0],kxy[0],kyy[0]))/0.00000001)
print(grad_MMD(p,n,m,grad_g,weights,k1yy,k1xy)[par])

#%%
#%%
iterations = 1000
#mses = mse(iterations,1,thetas,theta_star)[:,0]
plt.plot(range(iterations-1),mses[:,0])
plt.xlabel('SGD iterations',fontsize='x-large')
plt.ylabel('MSE of theta_1',fontsize='x-large')

plt.show
#%%
iterations = 1000
#mses = mse(iterations,1,thetas,theta_star)[:,0]
plt.plot(range(iterations-1),mses[:,1])
plt.xlabel('SGD iterations',fontsize='x-large')
plt.ylabel('MSE of theta_2',fontsize='x-large')

plt.show
#%%
iterations = 1000
#mses = mse(iterations,1,thetas,theta_star)[:,0]
plt.plot(range(iterations-1),mses[:,2])
plt.xlabel('SGD iterations',fontsize='x-large')
plt.ylabel('MSE of theta_3',fontsize='x-large')

plt.show
#%%
iterations = 1000
#mses = mse(iterations,1,thetas,theta_star)[:,0]
plt.plot(range(iterations-1),mses[:,3])
plt.xlabel('SGD iterations',fontsize='x-large')
plt.ylabel('MSE of theta_4',fontsize='x-large')

plt.show

#%%
#%%
def mse(max_it,p,thetas,theta_star):
    mse_ = np.zeros((max_it-1,p))
    for l in range(p):
        for j in range(max_it-1):
            mse_[j,l] = np.mean(np.asarray((thetas[1:j+2]-theta_star[l]))**2)
    return mse_


#%%
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
import plotly.figure_factory as ff
#fig = ff.create_distplot(sample[:,0], 'a')
fig = ff.create_distplot([sample[:,0],sample[:,1],sample[:,2],sample[:,3]], ['a','b','g','k'])
plot(fig)

#%%
def clean(serie):
    output = serie[(np.isnan(serie) == False) & (np.isinf(serie) == False)]
    return output

#%%
#np.savetxt(fname='15_March_results_gauss.txt', X=results)
#np.savetxt(fname='15_March_sample_gauss.txt', X=sample)
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
import plotly.figure_factory as ff
fig = ff.create_distplot([clean(sample[:,0]), clean(sample[:,1])], ['theta_1','theta_2'])
plot(fig)
#%%
import seaborn as sns  # xreiazetai ena sugekrimeno version des poio exeis kai kanto note
import matplotlib.pyplot as plt
#%%
#sample = np.loadtxt('sam.txt')

#%%
#%%
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X_ = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X_, Y = np.meshgrid(X_, Y)

# Mean vector and covariance matrix
mu = np.array([0., 1.])
Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X_.shape + (2,))
pos[:, :, 0] = X_
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X_, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X_, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)

plt.show()

#%%

#df = pd.read_csv('wabc_results_df.csv')
df = pd.read_csv('1d_wabc_200.csv')
theta1_abc = df["X1"].to_numpy()
#theta2_abc = df["X2"].to_numpy()
#thetas = [theta1_abc, theta2_abc]