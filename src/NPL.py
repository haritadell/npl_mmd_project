#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:28:49 2021

"""

from utils import k, k_jax
import itertools
from numpy.random import choice
import numpy as np
from scipy.stats import dirichlet
import scipy.spatial.distance as distance
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import stats
import ot
from scipy.optimize import minimize
import jax.numpy as jnp
from jax import grad, vmap, value_and_grad, jit
from jax.ops import index_update
from jax.config import config
from jax.experimental import optimizers


# NPL class
class npl():
    """This class contains functions to perform NPL inference for any of the models in models.py. 
    The user supplies parameters:
        X: Data set 
        B: number of bootstrap iterations
        m: number of points sampled from P_\theta at each optim. iteration
        s: standard deviation for Gaussian model
        p: number of unknown parameters
        l: lengthscale of gaussian kernel
        model: model in the form as in models.py
        model_name: name set to 'gaussian' or 'gandk' or 'toggle_switch'
        method_gd: gradient-based optimisation method set to eithr 'SGD' or 'NSGD'
    """
    
    def __init__(self, X, B, m, s, p, l, model, model_name = 'gaussian', method_gd = 'SGD'):
        self.model = model
        self.B = B  
        self.m = m  
        self.s = s  
        self.p = p  
        self.X = X  
        self.n, self.d = self.X.shape   
        self.method_gd = method_gd 
        self.model_name  = model_name 
        self.l = l  
        # set l = -1 for median heuristic 
        if self.l == -1:
            self.l = np.sqrt((1/2)*np.median(distance.cdist(self.X,self.X,'sqeuclidean')))
        self.kxx = k(self.X,self.X,self.l)[0] # pre calculate kernel matrix of data k(x,x)
    
        
        
    def draw_single_sample(self, seed):
        """ Draws a single sample from the nonparametric posterior specified via
        the model and the data X"""
        
        # draw Dirichlet weights
        weights = dirichlet.rvs(np.ones(self.n), size = 1, random_state = seed).flatten()  
       
        # compute weighted log-likelihood minimizer for Gaussian model
        if self.model_name == 'gaussian':
            wll_j = self.WLL(self.X, weights) 
            was_j = self.minimise_wasserstein(self.X, weights)
        else:
            wll_j = 1 # dummy variable 
            was_j = 1 # dummy variable
        
        # compute MMD minimizer
        if self.model_name == 'toggle_switch': # For toggle-switch model we use JAX 
            theta_j = self.minimise_MMD_togswitch(self.X, weights)
        else:
            theta_j = self.minimise_MMD(self.X, weights)
             
        return theta_j, wll_j, was_j
        
    
    def draw_samples(self):
        """Draws B samples in parallel from the nonparametric posterior"""
        
        # create objects to log the optimization results
        sample = np.zeros((self.B,self.p))
        wll = np.zeros((self.B,self.p))
        was_sample = np.zeros((self.B,self.p))
        
        # Parallelize
        temp = Parallel(n_jobs=-1, backend='multiprocessing', max_nbytes=None,batch_size="auto")(delayed(self.draw_single_sample)(i) for i in tqdm(range(self.B)))
    
        for i in range(self.B):
            sample[i,:] = temp[i][0]
            wll[i,:] = temp[i][1]
            was_sample[i,:] = temp[i][2]
            
        self.sample = np.array(sample)
        self.wll = np.array(wll)
        self.was = np.array(was_sample)
     
    
    
    def g_approx(self,grad_g, k2yy):
        """Approximation of the information metric for natural gradient descent"""
        
        grad_g_T = np.einsum('ijk -> jik',grad_g)
        prod1 = np.einsum('ijk, jlkm -> ilkm', grad_g_T, k2yy)
        prod2 = np.einsum('ijkl,jmk->imkl', prod1, grad_g)
        for i in range(self.p):
            np.fill_diagonal(prod2[i,i,:,:], 0)
        gsum = np.einsum('ijkl->ij', prod2)
    
        return 1/(self.m*(self.m-1))*gsum

    
    def MMD_approx(self,weights,kxy,kyy):
        """ Approximation of the squared MMD between the model represented by y_{i=1}^m iid sample and
        a sample from the DP posterior
        """
        
        # first sum
        np.fill_diagonal(kyy, np.repeat(0,self.m)) # excludes k(y_i, y_i) (diagonal terms)
        sum1 = np.sum(kyy)
    
        # second sum
        sum2 = np.sum(kxy)
    
        # third sum
        np.fill_diagonal(self.kxx, np.repeat(0,self.n))
        kxx = self.kxx*weights 
        sum3 = np.sum(kxx)
    
        return (1/(self.m*(self.m-1)))*sum1-(2/(self.m))*sum2+(1/(self.n-1))*sum3
        
    
    
    def grad_MMD(self,grad_g,k1yy,k1xy,batchsize):
        """Approximates the gradient of the MMD with respect to theta when the gradient of the generator is known"""
    
        # first sum
        prod1 = np.squeeze(np.einsum('ilj,imjk->lmjk', grad_g, np.expand_dims(k1yy,axis=1)))
        if prod1.ndim==2:
            np.fill_diagonal(prod1[:,:], 0)
            sum1 = np.sum(prod1)
        else:
            for i in range(self.p):
                np.fill_diagonal(prod1[i,:,:], 0)
            sum1 = np.einsum('ijk->i',prod1)

        # second sum
        prod2 = np.squeeze(np.einsum('ilj,imjk->lmjk', grad_g, np.expand_dims(k1xy,axis=1)))
        if prod2.ndim==2:
            sum2 = np.sum(prod2)
        else:
            sum2 = np.einsum('ijk->i',prod2)
    
        return (2/(self.m*(self.m-1)))*sum1-(2/(self.m*batchsize))*sum2
    
    def mmd_ts(self,theta,data,batchsize):
        """Function to approximate the squared MMD in jax for use in the toggle switch model"""
        
        sample = self.model.sample(theta)[0]
        
        kxx = k_jax(data,data,self.l) # different for each batch
        kyy = k_jax(sample,sample,self.l)
        kxy = k_jax(sample,data,self.l)
        
        # first sum
        diag_elements = jnp.diag_indices_from(kyy)
        kyy = index_update(kyy, diag_elements, jnp.repeat(0,self.m))
        sum1 = jnp.sum(kyy)
    
        # second sum
        sum2 = jnp.sum(kxy)
    
        # third sum
        diag_elements = jnp.diag_indices_from(kxx)
        kxx = index_update(kxx, diag_elements, jnp.repeat(0,batchsize))
        sum3 = jnp.sum(kxx)
            
        mmd = (1/(self.m*(self.m-1)))*sum1-(2/(self.m*batchsize))*sum2+(1/(batchsize*(batchsize-1)))*sum3
        
        return mmd
    
    def grad_mmd_ts(self,theta,data,batchsize):
        """Function for the gradient of the squared MMD using jax autodiff for use 
        in the toggle switch model"""
        
        #config.update("jax_debug_nans", True)  # check for nans in gradients - comment out after debugging
        config.update("jax_enable_x64", True)   # for stability
        gradient = grad(self.mmd_ts, argnums=0) 
        return gradient(theta,data,batchsize)
        
    def minimise_MMD(self, data, weights, Nstep=150, eta=0.1, batchsize=128): 
        """ Gradient-based optimization to minimize the MMD objective between 
        P^(j) for alpha=0 and P_\theta with
        Number of iterations = Nstep  
        Step size = eta
        Batch size = batchsize"""
        
        # Initialise
        if self.model_name == 'gandk':
            theta = np.zeros(self.p)
            theta[0] = np.median(self.X)
            theta[1] = stats.iqr(self.X, interpolation = 'midpoint')/2
            theta[2] = 0
            theta[3] = 0
        else:
            theta = 0.5*np.ones(self.p)
        t = 0
        
        while t < Nstep:  
            for i in range(self.n//batchsize):
                if self.d == 1:
                    batch_x = choice(data.flatten(), batchsize, p=weights).reshape(-1,1) # sample batch with replacement
                else:
                    idx = choice(range(len(data)),batchsize, p=weights)
                    batch_x = data[idx].reshape(batchsize,self.d)
                if self.model_name == 'gandk':
                    sample, z = self.model.sample(theta)
                    grad_g = self.model.grad_generator(z, theta)
                else:
                    sample = self.model.sample(theta)
                    grad_g = self.model.grad_generator(theta)
                
                # Calculate kernel matrices and gradients
                kyy, k1yy, k2yy = k(sample,sample,self.l)
                kxy, k1xy = k(sample,batch_x,self.l)[0:2]    
            
                # approximate MMD gradient
                if self.p == 1:
                    gradient = np.asmatrix(self.grad_MMD(grad_g,k1yy,k1xy,batchsize))
                else:
                    gradient = self.grad_MMD(grad_g,k1yy,k1xy,batchsize) 
                
                # approximate information metric
                
                # pre-define noise for information metric
                noise = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            
                if self.method_gd == 'NSGD':
                    g = self.g_approx(grad_g, k2yy)
                    # add noise if g can't be inverted
                    for j in range(9):
                        check = True
                        try:
                            np.linalg.inv(g + np.eye(self.p)*noise[j])
                        except np.linalg.LinAlgError:
                            check = False
                        if check:
                            break
                    g = g + np.eye(self.p)*noise[j]
            
                # update theta according to optimization method
                if self.method_gd == 'SGD':
                    theta = theta-eta*gradient
                else:
                    theta = theta-eta*np.linalg.inv(g)@gradient
            
            t += 1
        
        return theta  
    
    def minimise_MMD_togswitch(self, data, weights, Nstep=300, eta=0.6, batch_size=500):
        """Function to minimise the MMD for the toggle switch model using 
        adagrad optimisation from jax"""
        
        config.update("jax_enable_x64", True)
        num_batches = self.n//batch_size
        
        # objective function to feed the optimizer
        def obj_fun(theta, x, n):
            y = self.model.sample(theta)[0]
            kyy = k_jax(y,y,self.l)
            kxy = k_jax(y,x,self.l)

            # first sum
            diag_elements = jnp.diag_indices_from(kyy)
            kyy = index_update(kyy, diag_elements, jnp.repeat(0,self.m))
            sum1 = jnp.sum(kyy)
    
            # second sum
            sum2 = jnp.sum(kxy)
    
            return (1/(self.m*(self.m-1)))*sum1-(2/(n*self.m))*sum2 
        
        # initialisation
        params = jnp.array([20.,1.,3.,2.,320.,-2.,0.01])
        opt_init, opt_update, get_params = optimizers.adagrad(step_size=eta) 
        opt_state = opt_init(params)
        itercount = itertools.count()

        loss_and_grad_fn = vmap(value_and_grad(obj_fun, argnums=0), in_axes=(None, 0, None))
        
        def step(step, opt_state, batches):
            value, grads = loss_and_grad_fn(get_params(opt_state), batches, batch_size)
            opt_state = opt_update(step, np.mean(grads, axis=0), opt_state) 
            return value, opt_state
        
        for i in range(Nstep):
            batches = []
            # sample batches
            for _ in range(num_batches):
                batch_x = choice(data.flatten(), batch_size, p=weights).reshape(-1,1)
                batches.append(batch_x)
            batches = np.reshape(batches, (num_batches,batch_size))
            # update  
            value, opt_state = step(next(itercount), opt_state, batches)
            
        return get_params(opt_state)

    def minimise_MMD_ts(self, data, weights, Nstep=200, eta=0.1, batchsize=512): 
        """Gradient based optimisation to minimise the MMD using batch gradient descent 
        for the toggle switch model"""
        
        #initialise
        theta = np.array([20.,10.,2.,2.,300.,1.,1.]) 
        t = 0
        while t < Nstep:  
            for i in range(self.n//batchsize):
                batch_x = choice(data.flatten(), batchsize, p=weights).reshape(-1,1) # sample batches with replacement
                grad_ = self.grad_mmd_ts(theta,batch_x,batchsize)
                theta = theta - eta*grad_
                 
            print(grad_)
            
            t += 1
    
        return theta 
        
    def WLL(self, data, weights):
        """Get weighted log likelihood minimizer, for gaussian model""" 
        
        theta = 0
        for i in range(self.n):
            theta += weights[i]*data[i] 
        return theta
    
    def minimise_wasserstein(self, data, weights):
        """This function minimises the wasserstein distance 
        instead of the MMD using scipy optimizer Powell"""
        
        def wasserstein(theta):
            a = np.ones((self.m,)) / self.m 
            b = weights 
            sample = self.model.sample(theta)
         
            M = ot.dist(sample, data, 'euclidean')

            W = ot.emd2(a, b, M)
            return W
        
        optimization_result = minimize(wasserstein, np.mean(data,axis=0), method= 'Powell', options={'disp': False, 'maxiter': 10000})
     
        # return the value at optimum
        return optimization_result.x
    
