#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:57:30 2021

@author: HaritaDellaporta

Contains functions related to: Gaussian kernel and its derivatives, sampling from g-and-k and gaussian models using generators
"""

import numpy as np
import scipy.spatial.distance as distance

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

# Gaussian kernel  𝑘(𝑥,𝑦) , its gradient w.r.t. first element  ∇1𝑘(𝑥,𝑦)  and its second derivative w.r.t. to the second and first argument  ∇2∇1𝑘(𝑥,𝑦)
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

# Generator  𝐺𝜃(𝑢)  and generator gradient  ∇𝜃𝐺𝜃(𝑢)  for the Gaussian location model:
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
        outl = gen_gaussian(cont_size, d, unif_outl, 5*np.ones(d),s)
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
    unif_outl = np.random.rand(cont_size,d+1)
  
    # generate standard normals  
    z = normals(n_real,d,unif)
    z_outl = normals(cont_size, d, unif_outl)

    # generate samples from g-and-k distribution
    if n_cont != 0:
        outl = gen_gandk(z_outl, theta) + 15
        sample = gen_gandk(z,theta)
        x = np.concatenate((sample, outl), axis=0)
    else:
        x = gen_gandk(z,theta)
    
    #outl = np.asmatrix(np.random.normal(loc=5,scale=1,size=cont_size)).transpose()

    return np.asarray(x)   # need to return z as well?

