#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:51:49 2021

@author: HaritaDellaporta

Check if MMD gradient approximation is correct using finite difference
"""

from utils import sample_gaussian_outl, sample_gandk_outl, k, gen_gandk
import models
import numpy as np
from copy import deepcopy

#%%
def MMD_approx(n,m,weights,kxx,kxy,kyy):
    """y_{i=1}^m iid from P_\theta, x_{i=1}^n iid from P_0
    kxx is nxn, kyy is mxm and kxy is nxm"""
        
    # first sum
    np.fill_diagonal(kyy, np.repeat(0,m)) # exclude k(y_i, y_i) (diagonal terms)
    sum1 = np.sum(kyy)
    
    # second sum
    #sum2 = np.sum(kxy)
    sum2 = np.sum(kxy*weights)
    
    # third sum
    np.fill_diagonal(kxx, np.repeat(0,n))
    #sum3 = np.sum(self.kxx)
    sum3 = np.sum(kxx*weights)
    
    #return (1/(m*(m-1)))*sum1-(2/(m*n))*sum2+(1/(n*(n-1)))*sum3
    return (1/(m*(m-1)))*sum1-(2/(m))*sum2+(1/(n-1))*sum3

def grad_MMD(p,n,m,grad_g,weights,k1yy,k1xy):
    
    # first sum
    prod1 = np.squeeze(np.einsum('ilj,imjk->lmjk', grad_g, np.expand_dims(k1yy,axis=1)))
    if prod1.ndim==2:
        np.fill_diagonal(prod1[:,:], 0)
        sum1 = np.sum(prod1)
    else:
        for i in range(p):
            np.fill_diagonal(prod1[i,:,:], 0)
        sum1 = np.einsum('ijk->i',prod1)
    
    # second sum
    for i in range(1):
        k1xy[i,:,:] = k1xy[i,:,:]*weights
    prod2 = np.squeeze(np.einsum('ilj,imjk->lmjk', grad_g, np.expand_dims(k1xy,axis=1)))
    if prod2.ndim==2:
        sum2 = np.sum(prod2)
    else:
        sum2 = np.einsum('ijk->i',prod2)
    
    #return (2/(m*(m-1)))*sum1-(2/(n*m))*sum2
    return (2/(m*(m-1)))*sum1-(2/(m))*sum2


# Check gradient for gandk model 

def check_gandk_grad():
    m = 1000              # number of simulated samples
    n = 1500              # number of true samples
    theta = np.array([3,1,1,-np.log(2)])     # true theta
    d = 1
    p = 4                 # dimensions of data
    l = -1                # lengthscale (l=-1 for median heuristic)

    y, z = sample_gandk_outl(m,d,theta,n_cont=0)
    x, _ = sample_gandk_outl(n,d,theta,n_cont=0)

    kxx = k(x,x,l)
    kxy = k(y,x,l)
    kyy = k(y,y,l)
    k1yy = kyy[1]
    k1xy = kxy[1]
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
    
def check_gaussian_grad():
    m = 1000              # number of simulated samples
    n = 1500              # number of true samples
    theta = np.ones(2)    # true theta
    d = len(theta)        # dimensions of data
    p = d                 # dimensions of parameter space
    s = 1                # standard deviation of the model
    l = -1                # lengthscale (l=-1 for median heuristic)
    
    y = sample_gaussian_outl(m,d,s,theta)
    x = sample_gaussian_outl(n,d,s,theta)
    
    kxx = k(x,x,l)
    kxy = k(y,x,l)
    kyy = k(y,y,l)
    k1yy = kyy[1]
    k1xy = kxy[1]
    model = models.gauss_model(m,d,s)
    grad_g = model.grad_generator(theta)
    
    # check gradient using finite differences
    weights = (1/n)*np.ones(n)
    print('check gradient of the MMD^2 approximation:')
    print((MMD_approx(n,m,weights,kxx[0],k(y+0.00000001,x,l)[0],k(y+0.00000001,y+0.00000001,l)[0])-MMD_approx(n,m,weights,kxx[0],kxy[0],kyy[0]))/0.00000001)
    print(np.sum(grad_MMD(p,n,m,grad_g,weights,k1yy,k1xy)))
    
def MMD_sgd(p,data, weights, m, Nstep=1000 ,gamma=-1, eta=0.1, model, model_name="gaussian"):
    """Gaussian kernel of parameter gamma=1
    Minibatch size = sample size = n, number of iterations = Nstep = 1000
    MMD minimzer between P^(j) for \alpha=0 and P_\theta using PSGD
    """
    n, d = data.shape
    #print(d)
    theta = np.zeros(d)
    thetas = []
    grads = []
    losses = []
    t = 0
    # median heuristic if gamma=-1
    if gamma == -1:
        gamma = np.sqrt((1/2)*np.median(distance.cdist(data,data,'sqeuclidean')))
    kxx = k(data,data,gamma, sparse=True)[0]
    while t < Nstep: 
        if model_name == 'gandk':
            sample, z = model.sample(theta)
            grad_g = model.grad_generator(z, theta)
        else:
            sample = model.sample(theta)
            grad_g = model.grad_generator(theta)
                
        kyy, k1yy = k(sample,sample,gamma)[0:2]
        kxy, k1xy = k(sample,data,gamma)[0:2]    
        
        if p == 1:
            gradient = np.asmatrix(grad_MMD(d,n,m,grad_g,weights,k1yy,k1xy))
        else:
            gradient = grad_MMD(d,n,m,grad_g,weights,k1yy,k1xy)
        theta = theta-eta*gradient
        
        loss = MMD_approx(n,m,weights,kxx,kxy,kyy)
        losses.append(loss)
        thetas.append(theta)
        t += 1

    return np.squeeze(np.asarray(thetas)), losses, t

def mse(max_it,p,thetas,theta_star):
    mse_ = np.zeros((max_it-1,p))
    for l in range(p):
        for j in range(max_it-1):
            mse_[j,l] = np.mean(np.asarray((thetas[1:j+2]-theta_star[l]))**2)
    return mse_

def check_plot_mmd(model, model_name):
    """ check MMD optimisation step in a single bootstrap iteration"""
    # Set parameters 
    n=2**11
    m=2**9
    d=2
    s=1
    theta_star=np.ones(2)
    weights = dirichlet.rvs(np.ones(n), size = 1).flatten()
    X = sample_gaussian_outl(n,d,s,theta_star,n_cont=0)      
    # Optimise
    thetas, losses, iterations = MMD_sgd(X, weights, m)
    # Plot results
    mses = mse(iterations,1,thetas,theta_star)[:,0]
    
    fig = plt.figure(figsize=(10, 10))
    
    plt.subplot(1, 2, 1,)
    plt.plot(range(iterations-1),mses)
    plt.xlabel('SGD iterations',fontsize='x-large')
    plt.ylabel('MSE',fontsize='x-large')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(iterations), losses)
    plt.xlabel('SGD iterations', fontsize='x-large')
    plt.ylabel('MMD^2 loss', fontsize='x-large')

    return fig






    