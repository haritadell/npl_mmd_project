#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:28:49 2021

"""

from utils import k, k_jax, k_comp
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
from jax.ops import index_update, index
from jax.config import config
from jax.experimental import optimizers


# NPL class
class npl():
    """This class contains functions to perform NPL inference for any of the models in models.py. 
    The user supplies parameters:
        X: Data set 
        B: number of bootstrap iterations
        m: number of points sampled from P_\theta at each optim. iteration
        p: number of unknown parameters
        l: lengthscale of gaussian kernel
        model: model in the form as in models.py
        model_name: name set to 'gaussian' or 'gandk' or 'toggle_switch'
    """
    
    def __init__(self, X, B, m, p, l, model, model_name = 'gaussian'):
        self.model = model
        self.B = B  
        self.m = m  
        self.p = p  
        self.X = X  
        self.n, self.d = self.X.shape   
        self.model_name  = model_name 
        self.l = l  
        # set l = -1 for median heuristic 
        if self.l == -1:
            self.l = np.sqrt((1/2)*np.median(distance.cdist(self.X,self.X,'sqeuclidean')))
        self.kxx = k(self.X,self.X,self.l)[0] # pre calculate kernel matrix of data k(x,x)
        self.s = 1 # standard deviation for Gaussian model
    
        
        
    def draw_single_sample(self, weights):
        """ Draws a single sample from the nonparametric posterior specified via
        the model and the data X"""
        
        # compute weighted log-likelihood minimizer for Gaussian model
        if self.model_name == 'gaussian':
            wll_j = self.WLL(self.X, weights) 
            was_j = self.minimise_wasserstein(self.X, weights)
        else:
            wll_j = 1 # dummy variable 
            was_j = 1 # dummy variable
        
        if self.model_name == 'toggle_switch':
            theta_j = self.minimise_MMD_togswitch(self.X, weights)
        else:
            theta_j = self.minimise_MMD(self.X, weights)
             
        return theta_j #, wll_j, was_j
        
    
    def draw_samples(self):
        """Draws B samples in parallel from the nonparametric posterior"""
        
        # Draw weights for each bootstrap iteration and then parallelize using JAX vmap
        weights = dirichlet.rvs(np.ones(self.n), size = self.B, random_state = 11)
        samples = vmap(self.draw_single_sample, in_axes=0)(weights)
        self.sample = np.array(samples) 

    def MMD_approx(self,kxy,kyy):
        """ Approximation of the squared MMD between the model represented by y_{i=1}^m iid sample and
        a sample from the DP posterior
        """
        
        # first sum
        diag_elements = jnp.diag_indices_from(kyy)
        kyy = index_update(kyy, diag_elements, jnp.repeat(0,self.m))
        sum1 = jnp.sum(kyy)
    
        # second sum
        sum2 = jnp.sum(kxy)

        # third sum 
        diag_elements = jnp.diag_indices_from(self.kxx)
        kxx = index_update(self.kxx, diag_elements, jnp.repeat(0,self.n))
        sum3 = jnp.sum(kxx)
    
        return (1/(self.m*(self.m-1)))*sum1-(2/(self.n*self.m))*sum2+(1/(self.n*(self.n-1)))*sum3
        
    
    def minimise_MMD(self, data, weights, Nstep=1000, eta=0.1, batch_size=100):
        """Function to minimise the MMD using adam optimisation from jax"""
        
        params=jnp.zeros(self.p)
        if self.model_name == 'gandk':
            batch_size = 64
            params = jnp.array([5,5,5,5])
        
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
            
        
        opt_init, opt_update, get_params = optimizers.adam(step_size=eta) 
        opt_state = opt_init(params)
        itercount = itertools.count()

        grad_fn = vmap(jit(grad(obj_fun, argnums=0)), in_axes=(None, 0, None))
        
        def step(step, opt_state, batches):
            value, grads = grad_fn(get_params(opt_state), batches, batch_size)
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
    
    def minimise_MMD_togswitch(self, data, weights, Nstep=2000, eta=0.04, batch_size=2000):
        """Function to minimise the MMD for the toggle switch model using 
        adam optimisation from jax"""
        #batch_size = self.n
        config.update("jax_enable_x64", True)
        num_batches = self.n//batch_size
        
        n_optimized_locations = 3

        # objective function to feed the optimizer
        def obj_fun(theta, x, n):
            #y = self.model.sample(theta)[0]
            y = self.model.sample(theta)
            kyy = k_comp(y,y)
            kxy = k_comp(y,x)

            # first sum
            diag_elements = jnp.diag_indices_from(kyy)
            kyy = index_update(kyy, diag_elements, jnp.repeat(0,self.m))
            sum1 = jnp.sum(kyy)
    
            # second sum
            sum2 = jnp.sum(kxy)
    
            return (1/(self.m*(self.m-1)))*sum1-(2/(n*self.m))*sum2 
        

        opt_init, opt_update, get_params = optimizers.adam(step_size=eta) 
        itercount = itertools.count()

        grad_fn = vmap(jit(value_and_grad(obj_fun, argnums=0)), in_axes=(None, 0, None))
        
        def step(step, opt_state, batches):
            values, grads = grad_fn(get_params(opt_state), batches, batch_size)
            opt_state = opt_update(step, np.mean(grads, axis=0), opt_state) 
            value = np.mean(values, axis=0)
            return value, opt_state
        
        key = random.PRNGKey(11)

        list_of_thetas = jnp.zeros((n_optimized_locations,7))
        for j in range(n_optimized_locations):
          opt_state = opt_init(self.best_init_params[j,:])
          smallest_loss = 10000
          best_theta = get_params(opt_state)
          for i in range(Nstep):
            # get batches 
            batches = []
            for _ in range(num_batches):
              weights_shape = weights.reshape((self.n,1))
              key, subkey = jax.random.split(key)
              batch_x = jax.random.choice(key, a=data.flatten(), shape=(batch_size,1), p=weights).reshape(-1,1)
              batches.append(batch_x)

            batches = jnp.array(batches)
            batches = jnp.reshape(batches, (num_batches,batch_size))
            # update  
            value, opt_state = step(next(itercount), opt_state, batches)
            pred =  value < smallest_loss
            def true_func(args):
              value, smallest_loss, best_theta, opt_state = args[0], args[1], args[2], args[3]
              smallest_loss = value
              best_theta = get_params(opt_state)
              return smallest_loss, best_theta 
            def false_func(args):
              value, smallest_loss, best_theta, opt_state = args[0], args[1], args[2], args[3]
              smallest_loss = jnp.array(smallest_loss, dtype='float64')
              return smallest_loss, best_theta
            smallest_loss, best_theta = jax.lax.cond(pred, true_func, false_func, [value, smallest_loss, best_theta, opt_state])

          list_of_thetas = index_update(list_of_thetas, index[j,:], best_theta)

        loss_min = 10000
        
        losses = []
        seed = 12
        rng = jax.random.PRNGKey(seed)
        rng, *rng_inputs = jax.random.split(rng, num=n_optimized_locations + 1)
        for t in list_of_thetas:
          losses.append(self.loss(rng,t))

        best_theta = list_of_thetas[jnp.argmin(jnp.asarray(losses))]
        
        return best_theta
        
        
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
    
