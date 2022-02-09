#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:57:30 2021


Contains functions related to: Gaussian kernel and its derivatives, sampling 
from gaussian, g-and-k and toggle switch models using generators, MMD approximation function 
and computation of MSE
"""

import numpy as np
from jax import numpy as jnp
from jax import vmap
import scipy.spatial.distance as distance
from scipy import stats
import math
import copy
from numpy.random import choice
from sklearn.utils import shuffle

# Box-Muller transformation
def boxmuller(unif1,unif2):
    u1 = np.sqrt(-2*np.log(unif1))*np.cos(2*np.pi*unif2)
    u2 = np.sqrt(-2*np.log(unif1))*np.sin(2*np.pi*unif2)
    return np.transpose(np.vstack([u1,u2]))

# Function generating standard normals using the box-muller transformation:
def normals(n, d, unif, sv=False):

    # avoid origin
    unif[unif==0] = np.nextafter(0, 1)

    # if d is odd, add one dimension
    if d % 2 != 0:
        dim = d + 1
    else:
        dim = d

    # expand dimensions for SV model
    if sv == True:
        dim = 2+2*d

    # create standard normal samples
    u = np.zeros((n,dim))
    for i in np.arange(0,dim,2):
        u[:,i:(i+2)] = boxmuller(unif[:,i],unif[:,(i+1)])

    # if d is odd, drop one dimension
    if d % 2 != 0 or sv == True:
        u = np.delete(u,-1,1)

    return u

# Gaussian kernel, its gradient w.r.t. first element and its second derivative w.r.t. to the second and first argument  
def k(x,y,l, sparse=False): 

    if sparse == True:
        x = x.astype('float32')
        y = y.astype('float32')
    
    # dimensions
    d = x.shape[1]
    dims = np.arange(d)
    
    # kernel
    kernel = np.exp(-(1/(2*l**2))*distance.cdist(x,y,'sqeuclidean'))
    
    if type(x) != np.ndarray: # using np.matrix.A to convert to ndarray
        x = x.A
    if type(y) != np.ndarray:
        y = y.A
    
    # first derivative
    grad_1 = -1*np.squeeze(np.subtract.outer(x,y)[:,[dims],:,[dims]], axis=0)*(1/l**2)*np.expand_dims(kernel, axis=0)
    
    #second derivative
    grad_21 = (1/l**2)*(np.expand_dims(np.expand_dims(np.eye(d), axis = 2), axis = 3)-np.einsum('ijk,ljk->iljk',np.squeeze(np.subtract.outer(x,y)[:,[dims],:,[dims]], axis=0),np.squeeze(np.subtract.outer(x,y)[:,[dims],:,[dims]], axis=0))*(1/l**2))*kernel
    
    return list([kernel, grad_1, grad_21])

def sqeuclidean_distance(x, y):
    return jnp.sum((x-y)**2)

def rbf_kernel(x, y, l):
    return jnp.exp( -(1/(2*l**2)) * sqeuclidean_distance(x, y))

def k_jax(x,y,l): 
    """Gaussian kernel compatible with JAX library"""

    x = x.astype('float64')
    y = y.astype('float64')
    mapx1 = vmap(lambda x, y: rbf_kernel(x, y, l), in_axes=(0, None), out_axes=0)
    mapx2 = vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)

    # kernel
    K = mapx2(x, y)
    
    return K

def k_comp(x,y):
    l_range = np.array([1.0,10.0,20.0,40.0,80.0,100.0,130.0,200.0,400.0,800.0,1000.0])
    n = len(x)
    m = len(y)
    k_gaus = np.zeros((n,m))
    for l in l_range:
      k_gaus += k_jax(x,y,l)
 
    return k_gaus 

# Function to get mini-batches for mini-batch gradient descent 
def get_batches(X, batchSize): 
    np.random.shuffle(X)
    for i in np.arange(0, X.shape[0], batchSize):
        yield X[i:i + batchSize]

# Generator and generator gradient for the Gaussian location model:
def gen_gaussian(n, d, unif, theta, sigma):

    unif[unif==0] = np.nextafter(0, 1)

    # if d is odd, add one dimension
    if d % 2 != 0:
        dim = d + 1
    else:
        dim = d

    # create standard normal samples
    u = np.zeros((n,dim))
    for i in np.arange(0,dim,2):
        u[:,i:(i+2)] = boxmuller(unif[:,i],unif[:,(i+1)])

    # if d is odd, drop one dimension
    if d % 2 != 0:
        u = np.delete(u,-1,1)

    # generate samples
    x = theta + u*sigma

    return x


# Function to sample from Gaussian location model with a percentage of outliers
    
def sample_gaussian_outl(n,d,s,theta,n_cont = 0):   # set n_cont to zero for no outliers
    
    cont_size = int(np.floor(int(n_cont)*5/100*n))
    n_real = n - cont_size
    
    # odd number of parameters
    if d % 2 != 0: 
        unif = np.random.rand(n_real,d+1)
        unif_outl = np.random.rand(cont_size, d+1)

    # even number of parameters
    else: 
       unif = np.random.rand(n_real,d)
       unif_outl = np.random.rand(cont_size,d)
    # use generator  
    
    if n_cont != 0:
        outl = gen_gaussian(cont_size, d, unif_outl, 20*np.ones(d),s)
        sample_ = gen_gaussian(n_real,d,unif,theta,s) 
        x = np.concatenate((sample_, outl), axis=0)
    else:
        x = gen_gaussian(n_real,d,unif,theta,s)  
    return x

# generator
def gen_gandk(z, theta):
    a = theta[0]
    b = theta[1]
    g = theta[2]
    k = np.exp(theta[3])
    g = a+b*(1+0.8*((1-np.exp(-g*z))/(1+np.exp(-g*z))))*((1+z**2)**(k))*z
    return g


def sample_gandk_outl(n,d,theta, n_cont = 0):
    
    cont_size = int(np.floor(int(n_cont)*5/100*n))
    n_real = n - cont_size

    # generate uniforms
    unif = np.random.rand(n_real,d+1)
    unif_outl1 = np.random.rand(int(cont_size//2),d+1)
    unif_outl2 = np.random.rand(int(cont_size - cont_size//2),d+1)
  
    # generate standard normals  
    z = normals(n_real,d,unif)
    z_outl1 = normals(int(cont_size//2), d, unif_outl1)
    z_outl2 = normals(int(cont_size - cont_size//2), d, unif_outl2)

    # generate samples from g-and-k distribution
    if n_cont != 0:
        outl1 = gen_gandk(z_outl1, theta) - 50
        outl2 = gen_gandk(z_outl2, theta) + 50
        sample = gen_gandk(z,theta)
        x = np.concatenate((sample, outl1, outl2), axis=0)
        x = shuffle(x)
    else:
        x = gen_gandk(z,theta)
    
    return np.asarray(x) 
    


    
def sample_togswitch_noise(params,n,T, df, add_noise=True):
    alpha1 = np.exp(params[0])
    alpha2 = np.exp(params[1])
    beta1 = params[2]
    beta2 = params[3]
    mu = np.exp(params[4])
    sigma = np.exp(params[5])
    gamma =  params[6]
  
    nsamples= n
    u = np.zeros((nsamples,T))
    v = np.zeros((nsamples,T))
    u_new = np.zeros((nsamples,T))
    v_new = np.zeros((nsamples,T))
    
    u[:,0] = 10.
    v[:,0] = 10.

    for t in range(0,T-1):

        u_new = u[:,t] +(alpha1/(1.+(v[:,t]**beta1)))-(1.+0.03*u[:,t])
        u[:,t+1] = u_new+0.5*stats.truncnorm.rvs(-2*u_new,math.inf, size=nsamples)

        v_new = v[:,t] +(alpha2/(1.+(u[:,t]**beta2)))-(1.+0.03*v[:,t])
        v[:,t+1] = v_new+0.5*stats.truncnorm.rvs(-2*v_new, math.inf, size=nsamples)

    lb = -(u[:,T-1] + mu) / (mu*sigma)*(u[:,T-1]**gamma)

    perc_noisy = 0.1
    n_noisy = int(perc_noisy*n)
    noise = stats.t.rvs(df, loc=0, scale=10, size=n_noisy)
    
    if add_noise == True:
      y = u[:,T-1] + mu + mu*sigma*stats.truncnorm.rvs(lb, math.inf, size=nsamples) / (u[:,T-1]**gamma)
      yvals = copy.deepcopy(y)
      y_noisy_index = choice(n, n_noisy)
      
      yvals[y_noisy_index] += noise
    else:
      yvals = u[:,T-1] + mu + mu*sigma*stats.truncnorm.rvs(lb, math.inf, size=nsamples) / (u[:,T-1]**gamma)
      y = yvals

    return np.atleast_2d(yvals), noise, y

def mse(theta,theta_star):
    mse_ = np.mean(np.asarray((theta-theta_star))**2)/np.mean(theta_star)
    return mse_

def MMD_approx(n,m,kxx,kxy,kyy):
    """ Approximates the squared MMD between P and Q given the gram matrices between samples y_{1:m} iid from P and x_{1:n} iid from Q
    """
        
    # first sum
    np.fill_diagonal(kyy, np.repeat(0,m)) # excludes k(y_i, y_i) (diagonal terms)
    sum1 = np.sum(kyy)
    
    # second sum
    sum2 = np.sum(kxy)
    
    # third sum
    np.fill_diagonal(kxx, np.repeat(0,n))
    sum3 = np.sum(kxx)
    
    return (1/(m*(m-1)))*sum1-(2/(n*m))*sum2+(1/(n*(n-1)))*sum3
    