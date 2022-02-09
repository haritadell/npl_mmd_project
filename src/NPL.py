#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:28:49 2021

"""

from utils import k, k_jax, k_comp
import itertools
import numpy as np
from scipy.stats import dirichlet
import scipy.spatial.distance as distance
from tqdm import tqdm
from joblib import Parallel, delayed
import ot
from scipy.optimize import minimize
import jax.numpy as jnp
import jax
from jax import vmap, value_and_grad, jit
from jax.ops import index_update, index
from jax.config import config
from jax.experimental import optimizers


# NPL class
class npl():
    """This class contains functions to perform NPL inference (for alpha = 0 in the DP prior) for any of the models in models.py. 
    The user supplies parameters:
        X: Data set 
        B: number of bootstrap iterations
        m: number of points sampled from P_\theta at each optim. iteration
        p: number of unknown parameters
        l: lengthscale of gaussian kernel; set l = -1 to use median heuristic
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
        if self.l == -1: # median heuristic
            self.l = np.sqrt((1/2)*np.median(distance.cdist(self.X,self.X,'sqeuclidean')))
        if self.model_name == 'toggle_switch':
            self.kxx = k_comp(self.X,self.X)
        else:
            self.kxx = k(self.X,self.X,self.l)[0] # pre calculate kernel matrix of data k(x,x)
        self.s = 1 # standard deviation for Gaussian model
    
        
        
    def draw_single_sample(self, weights):
        """ Draws a single sample from the nonparametric posterior specified via
        data X and Dirichlet weights"""
        
        # compute weighted log-likelihood minimizer for Gaussian model
        if self.model_name == 'gaussian':
            wll_j = self.WLL(self.X, weights) 
        else:
            wll_j = 1 # dummy variable 
           
        if self.model_name == 'toggle_switch':
            theta_j = self.minimise_MMD_togswitch(self.X, weights)
        else:
            theta_j = self.minimise_MMD(self.X, weights)
             
        return theta_j , wll_j 
    
    def loss(self, rng, theta):
        """Given parameter theta it approximates the MMD loss between P_theta and empirical measure P_n"""
        
        y = self.model.sample(theta)
        kxy = k_comp(y,self.X)
        kyy = k_comp(y,y)
        l = self.MMD_approx(kxy,kyy)
        return l
    

    def find_initial_params(self):
        """Function to find optimisation starting point for the toggle switch model"""
        
        def sample_theta_init(rng):
          param_range = (jnp.array([0.01, 0.01, 0.01, 0.01, 250.0, 0.01, 0.01]), jnp.array([50.0, 50.0, 5.0, 5.0, 450.0, 0.5, 0.4]))
          lower, upper = param_range
          unparam_theta = jax.random.uniform(rng, minval=lower, maxval=upper, shape=lower.shape)
          params = jnp.array([jnp.log(unparam_theta[0]),jnp.log(unparam_theta[1]),unparam_theta[2],unparam_theta[3],jnp.log(unparam_theta[4]),jnp.log(unparam_theta[5]),unparam_theta[6]])
          return params  

        n_initial_locations = 500
        n_optimized_locations = 3

        rng = jax.random.PRNGKey(2)
        rng, *rng_inputs = jax.random.split(rng, num=n_initial_locations + 1)
        init_thetas = vmap(sample_theta_init)(jnp.array(rng_inputs))
        rng, *rng_inputs = jax.random.split(rng, num=n_initial_locations + 1)

        init_losses = []
        for t in init_thetas:
          init_losses.append(self.loss(rng,t))

        rng, *rng_inputs = jax.random.split(rng, num=n_optimized_locations + 1)
        best_init_params = init_thetas[np.argsort(np.asarray(init_losses))[:n_optimized_locations]]

        return best_init_params 
        
    
    def draw_samples(self):
        """Draws B samples in parallel from the nonparametric posterior"""
        
        # Draw weights for each bootstrap iteration and then parallelize using JAX vmap
        if self.model_name == 'toggle_switch':
            self.best_init_params = self.find_initial_params()
        weights = dirichlet.rvs(np.ones(self.n), size = self.B, random_state = 13)
        was_samples = np.zeros((self.B,self.p))
        

        # For the Gaussian model also run NPL with the Wasserstein distance
        if self.model_name == 'gaussian':
            temp = Parallel(n_jobs=-1, backend='multiprocessing', max_nbytes=None,batch_size="auto")(delayed(self.minimise_wasserstein)(self.X,weights[i,:]) for i in tqdm(range(self.B)))

            for i in range(self.B):
                was_samples[i,:] = temp[i]
                self.was_sample = np.array(was_samples)

        samples, wll_samples = vmap(self.draw_single_sample, in_axes=0)(weights)
        self.sample = np.array(samples) 
        self.wll_sample = np.array(wll_samples)
        



    def MMD_approx(self,kxy,kyy):
        """ Approximation of the squared MMD given gram matrices kxy and kyy"""
        
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
        
    
    def minimise_MMD(self, data, weights, Nstep=1000, eta=0.1, batch_size=200):
        """Function to minimise the MMD using adam optimisation in JAX -- use this function for 
        the Gaussian and G-and-k models"""
        
        params = jnp.ones(self.p)
        if self.model_name == 'gandk':
            batch_size = self.n
            params = jnp.array([5.,5.,5.,5.])
        
        config.update("jax_enable_x64", True)
        num_batches = self.n//batch_size
        
        # objective function to feed the optimizer
        def obj_fun(theta, x, n, key):
            if self.model_name == 'gaussian':
                y = self.model.sample(theta)
            else:
                y = self.model.sample(theta,key)[0]
            kyy = k_jax(y,y, self.l)
            kxy = k_jax(y,x, self.l)

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

        #grad_fn = vmap(jit(grad(obj_fun, argnums=0)), in_axes=(None, 0, None, None))
        grad_fn = vmap(jit(value_and_grad(obj_fun, argnums=0)), in_axes=(None, 0, None, None))
        
        def step(step, opt_state, batches, key):
            key, subkey = jax.random.split(key)
            values, grads = grad_fn(get_params(opt_state), batches, batch_size, subkey)
            opt_state = opt_update(step, np.mean(grads, axis=0), opt_state) 
            value = np.mean(values, axis=0)
            return value, opt_state
        
        key1 = jax.random.PRNGKey(11)
        key2 = jax.random.PRNGKey(13)
        
        smallest_loss = 1000000
        best_theta = get_params(opt_state)
        for i in range(Nstep):            
            batches = []
            for _ in range(num_batches):
              key1, subkey = jax.random.split(key1)
              inds = jax.random.choice(subkey, a=self.n, shape=(batch_size,), p=weights) #default is with replacement
              batch_x = jnp.take(a=data, indices=inds, axis=0)
              batches.append(batch_x)

            batches = jnp.array(batches)
            # update  
            value, opt_state = step(next(itercount), opt_state, batches, key2)
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
            
        return best_theta
    
    def minimise_MMD_togswitch(self, data, weights, Nstep=2000, eta=0.04, batch_size=2000):
        """Function to minimise the MMD for the Toggle switch model using 
        Adam optimisation in JAX"""
        
        config.update("jax_enable_x64", True)
        num_batches = self.n//batch_size
        
        n_optimized_locations = 3

        # objective function to feed the optimizer
        def obj_fun(theta, x, n):
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
        
        key = jax.random.PRNGKey(11)

        list_of_thetas = jnp.zeros((n_optimized_locations,7))
        for j in range(n_optimized_locations):
          opt_state = opt_init(self.best_init_params[j,:])
          smallest_loss = 10000
          best_theta = get_params(opt_state)
          for i in range(Nstep):
            # get batches 
            batches = []
            for _ in range(num_batches):
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
              _, smallest_loss, best_theta, _ = args[0], args[1], args[2], args[3]
              smallest_loss = jnp.array(smallest_loss, dtype='float64')
              return smallest_loss, best_theta
            smallest_loss, best_theta = jax.lax.cond(pred, true_func, false_func, [value, smallest_loss, best_theta, opt_state])

          list_of_thetas = index_update(list_of_thetas, index[j,:], best_theta)
        
        losses = []
        seed = 12
        rng = jax.random.PRNGKey(seed)
        rng, *rng_inputs = jax.random.split(rng, num=n_optimized_locations + 1)
        for t in list_of_thetas:
          losses.append(self.loss(rng,t))

        best_theta = list_of_thetas[jnp.argmin(jnp.asarray(losses))]
        
        return best_theta
        
        
    def WLL(self, data, weights):
        """Get weighted log likelihood minimizer, for Gaussian location model""" 
        
        theta = np.zeros(self.d)
        print(np.shape(weights))
        for i in range(self.n):
            theta += weights[i]*data[i,:] 
        return theta
    
    def minimise_wasserstein(self, data, weights):
        """This function minimises the Wasserstein distance 
        instead of the MMD using Scipy optimizer Powell"""
        
        def wasserstein(theta):
            a = np.ones((self.m,)) / self.m 
            b = weights 
            sample = self.model.sample(theta)
         
            M = ot.dist(sample, data, 'euclidean')

            W = ot.emd2(a, b, M)
            return W
        
        optimization_result = minimize(wasserstein, np.zeros(self.p), method= 'Powell', options={'disp': True, 'maxiter': 20000})
        print(optimization_result.x)
        # return the value at optimum
        return optimization_result.x
    
