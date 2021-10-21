#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:19:23 2021

"""
from utils import boxmuller, normals, k_comp
import math
import numpy as np
from jax import grad, lax, jit, vmap, random
import jax.numpy as jnp
from jax.scipy import stats as jstats
from jax.ops import index, index_update

class Model():
    """An empty/abstract class that dictates the necessary functions a model type class 
    should have"""
    
    def __init__(self, m, params):
        self.m = m  # number of points sampled from the model at each optim. iteration
        self.params = params # hyperparameters relevant to each model
        
    def generetor(self, u, theta):
        """Generates samples from the simulator for parameter theta after 
        providing iid samples u"""
        return 0
    
    def grad_generator(self, u, theta):
        """Gradient of the generator with respect to theta"""
        return 0
        
    def sample(self, theta):
        """Given parameter theta returns m samples from the generator"""
        return 0 

class gauss_model():
    
    def __init__(self, m, d, s):
        self.m = m  # number of points sampled from P_\theta at each optim. iteration
        self.d = d  # data dim
        self.s = s  # standard deviation of the model
        
    def generator(self, unif, theta):

        unif[unif==0] = np.nextafter(0, 1)

        # if d is odd, add one dimension
        if self.d % 2 != 0:
            dim = self.d + 1
        else:
            dim = self.d

        # create standard normal samples
        u = np.zeros((self.m,dim))
        for i in np.arange(0,dim,2):
            u[:,i:(i+2)] = boxmuller(unif[:,i],unif[:,(i+1)])

        # if d is odd, drop one dimension
        if self.d % 2 != 0:
            u = np.delete(u,-1,1)

       # generate samples
        x = theta + u*self.s

        return x

    # gradient of the generator
    def grad_generator(self,theta):
        return np.broadcast_to(np.expand_dims(np.eye(theta.shape[0]),axis=2),(theta.shape[0],theta.shape[0],self.m))
    
    def sample(self,theta):

        # odd number of parameters
        if self.d % 2 != 0: 
            unif = np.random.rand(self.m,self.d+1)
        # even number of parameters
        else: 
            unif = np.random.rand(self.m,self.d)
        
        # use generator  
        x = self.generator(unif,theta)

        return x
    
class g_and_k_model():
    
    def __init__(self, m, d):
        self.d = d  # data dimension
        self.m = m  # number of points sampled from P_\theta at each optim. iteration
    
    # generator
    def generator(self, z, theta):
        a = theta[0]
        b = theta[1]
        g = theta[2]
        k = jnp.exp(theta[3])
        g = a+b*(1+0.8*((1-jnp.exp(-g*z))/(1+jnp.exp(-g*z))))*((1+z**2)**(k))*z
        return g

    # gradient of the generator
    def grad_generator(self, z,theta):
        #a = theta[0]
        b = theta[1]
        g = theta[2]
        k = np.exp(theta[3])
        grad1 = np.ones(z.shape[0])
        grad2 = (1+(4/5)*((1-np.exp(-g*z))/(1+np.exp(-g*z))))*(np.exp(k*np.log(1+z**2)))*z
        grad3 = (8/5)*theta[1]*((np.exp(g*z))/(1+np.exp(g*z))**2)*(np.exp(k*np.log(1+z**2)))*z**2
        grad4 = b*(1+0.8*((1-np.exp(-g*z))/(1+np.exp(-g*z))))*(np.exp(k*np.log(1+z**2)))*np.log(1+z**2)*z
        return np.expand_dims(np.einsum('ij->ji',np.c_[grad1,grad2,grad3,grad4]), axis=0)
    
    def sample(self,theta,key):          

      # generate uniforms
      #unif = np.random.rand(self.m,self.d+1)
      
      # generate standard normals  
      #z = normals(self.m,self.d,unif)
      z = random.normal(key, shape=(512,1))

      # generate samples from g-and-k distribution
      x = self.generator(z,theta)
  
      return jnp.asarray(x), jnp.asarray(z)
  
class toggle_switch_model():
    
    def __init__(self, m, d, T):
        self.d = d  # data dim
        self.m = m  # number of points sampled from P_\theta at each optim. iteration
        self.T = T
        self.seed = 11
        
    def ugenerator(self):
        uvals = np.random.uniform(size=(self.m,(2*self.T)+1))
        return(uvals)
        
    def generator(self,theta):

        alpha1 = jnp.exp(theta[0])
        alpha2 = jnp.exp(theta[1])
        beta1 = theta[2]
        beta2 = theta[3]
        mu = jnp.exp(theta[4])
        sigma = jnp.exp(theta[5])
        gamma = theta[6]

        nsamples= 500
        T = 300
        u = jnp.zeros(nsamples)
        v = jnp.zeros(nsamples)

     
        seed=13
        key = random.PRNGKey(seed)
        key, *key_inputs = random.split(key, num=T+1)

        u = index_update(u, index[:], 10.)
        v = index_update(v, index[:], 10.)
        init_list = jnp.array([u,v])

        def step(current_array,key):
            u_t, v_t = current_array[0], current_array[1]
            key, subkey = random.split(key)
            u_new = u_t +(alpha1/(1.+(v_t**beta1)))-(1.+0.03*u_t)
            u_next = u_new+0.5*random.truncated_normal(subkey,-2*u_new, math.inf)
            v_new = v_t +(alpha2/(1.+(u_t**beta2)))-(1.+0.03*v_t)
            key, subkey = random.split(key)
            v_next = v_new+0.5*random.truncated_normal(subkey,-2*v_new, math.inf)
            return jnp.array([u_next,v_next]), key

        final_array, _  = lax.scan(step, init_list, jnp.array(key_inputs))
        u, v = final_array[0], final_array[1]

        lb = -(u + mu) / (mu*sigma)*(u**gamma)
        key, subkey = random.split(key)
        yvals = u + mu + mu*sigma*random.truncated_normal(subkey, lb, math.inf, shape=(nsamples,)) / (u**gamma)

        return (jnp.atleast_2d(yvals).T)
        
        
    def sample(self,theta):
        #uvals = self.ugenerator()
        x = self.generator(theta) 
        return jnp.array(x)
    
    def generator_single(self,theta,uvals):
        """This function is used to find the gradient of the generator using JAX autodiff via vmap. 
        The user provides the value of the parameter theta and the one dimensional uniform samples uvals of length 2T+1, """
        
        alpha1 = theta[0]
        alpha2 = theta[1]
        beta1 = theta[2]
        beta2 = theta[3]
        mu = theta[4]
        sigma = theta[5]
        gamma =  theta[6]

        u = jnp.zeros(self.T, dtype='float64')
        v = jnp.zeros(self.T, dtype='float64')
        u_new = jnp.zeros(self.T)
        v_new = jnp.zeros(self.T)
        phi_u_new = jnp.zeros(self.T)
        phi_v_new = jnp.zeros(self.T)


        u = index_update(u, index[0], 10.)
        v = index_update(v, index[0], 10.)

        for t in range(0,self.T-1):

            u_new = u[t] +(alpha1/(1.+(v[t]**beta1)))-(1.+0.03*u[t])
            phi_u_new = jstats.norm.cdf(-2.*u_new)
            u = index_update(u, index[t+1],u_new+0.5*jstats.norm.ppf(phi_u_new+uvals[t]*(1.-phi_u_new)))

            v_new = v[t] +(alpha2/(1.+(u[t]**beta2)))-(1.+0.03*v[t])
            phi_v_new = jstats.norm.cdf(-2.*v_new)
            v = index_update(v, index[t+1], v_new+0.5*jstats.norm.ppf(phi_v_new+uvals[self.T+t]*(1.-phi_v_new)))


        lb = -(u[self.T-1] + mu) / (mu*sigma)*(u[self.T-1]**gamma)
        phi_lb = jstats.norm.cdf(lb)
        yval = (jstats.norm.ppf(phi_lb+uvals[2*self.T]*(1.-phi_lb))*(sigma)*(mu)*(u[self.T-1]**(-gamma)))+(mu+u[self.T-1]) 

        return yval
    
    def grad_generator(self,uvals,theta):  # automatic differentiation for gradient of generator using JAX
        gradient = grad(self.generator_single, argnums=0)
        grad_ = vmap(jit(gradient), in_axes=(None,0), out_axes=1)(theta,uvals)
        return jnp.reshape(grad_, (1,len(theta),self.m))
    
    
