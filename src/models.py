#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:19:23 2021

"""

import numpy as np
from scipy import stats

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
        self.d = d  # data dim
        self.m = m  # number of points sampled from P_\theta at each optim. iteration
    
    # generator
    def generator(self, z, theta):
        a = theta[0]
        b = theta[1]
        g = theta[2]
        k = np.exp(theta[3])
        g = a+b*(1+0.8*((1-np.exp(-g*z))/(1+np.exp(-g*z))))*((1+z**2)**(k))*z
        return g

    # gradient of the generator
    def grad_generator(self, z,theta):
        a = theta[0]
        b = theta[1]
        g = theta[2]
        k = np.exp(theta[3])
        grad1 = np.ones(z.shape[0])
        grad2 = (1+(4/5)*((1-np.exp(-g*z))/(1+np.exp(-g*z))))*(np.exp(k*np.log(1+z**2)))*z
        grad3 = (8/5)*theta[1]*((np.exp(g*z))/(1+np.exp(g*z))**2)*(np.exp(k*np.log(1+z**2)))*z**2
        grad4 = b*(1+0.8*((1-np.exp(-g*z))/(1+np.exp(-g*z))))*(np.exp(k*np.log(1+z**2)))*np.log(1+z**2)*z
        return np.expand_dims(np.einsum('ij->ji',np.c_[grad1,grad2,grad3,grad4]), axis=0)
    
    def sample(self,theta):          

      # generate uniforms
      
      unif = np.random.rand(self.m,self.d+1)
      
      # generate standard normals  
      z = normals(self.m,self.d,unif)

      # generate samples from g-and-k distribution
      x = self.generator(z,theta)
  
      return np.asarray(x), np.asarray(z)
  
class toggle_switch_model():
    
    def __init__(self, m, d, T):
        self.d = d  # data dim
        self.m = m  # number of points sampled from P_\theta at each optim. iteration
        self.T = T
        
    def ugenerator(self):
        uvals = np.random.uniform(size=(self.m,(2*self.T)+1))
        return(uvals)
        
    def generator(self,theta,uvals):

        alpha1 = theta[0]
        alpha2 = theta[1]
        beta1 = theta[2]
        beta2 = theta[3]
        mu = theta[4]
        sigma = theta[5]
        gamma =  theta[6]

        nsamples= uvals.shape[0]
        u = np.zeros((nsamples,self.T))
        v = np.zeros((nsamples,self.T))
        u_new = np.zeros((nsamples,self.T))
        v_new = np.zeros((nsamples,self.T))
        phi_u_new = np.zeros((nsamples,self.T))
        phi_v_new = np.zeros((nsamples,self.T))


        u[:,0] = 10.
        v[:,0] = 10.

        for t in range(0,self.T-1):

            u_new = u[:,t] +(alpha1/(1.+(v[:,t]**beta1)))-(1.+0.03*u[:,t])
            phi_u_new = stats.norm.cdf(-2.*u_new)
            u[:,t+1] = u_new+0.5*stats.norm.ppf(phi_u_new+uvals[:,t]*(1.-phi_u_new))

            v_new = v[:,t] +(alpha2/(1.+(u[:,t]**beta2)))-(1.+0.03*v[:,t])
            phi_v_new = stats.norm.cdf(-2.*v_new)
            v[:,t+1] = v_new+0.5*stats.norm.ppf(phi_v_new+uvals[:,self.T+t]*(1.-phi_v_new))


        yvals = (stats.norm.ppf(0.5+0.5*uvals[:,2*self.T])*(sigma**2)*(mu**2)*(u[:,self.T-1]**(-2.*gamma)))+(mu+u[:,self.T-1]) 

        return(np.atleast_2d(yvals).T)
        
    def sample(self,theta):
        uvals = self.ugenerator()
        x = self.generator(theta,uvals)
        return np.asarray(x)
    
    