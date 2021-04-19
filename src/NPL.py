#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:28:49 2021

"""

from utils import k
import numpy as np
from scipy.stats import dirichlet
import scipy.spatial.distance as distance
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy import stats
import ot
from autograd import grad

# NPL class
class npl():
    """This class contains functions to perform NPL inference for any of the models in models.py
    
    Parameters:
        X: Data set 
        B: number of bootstrap iterations
        m: number of points sampled from P_\theta at each optim. iteration
        s: standard deviation for Gaussian model
        p: number of unknown parameters
        l: lengthscale of gaussian kernel
        model: model in the form as in models.py
        model-name: name set to 'gaussian' or 'gandk' or 'toggle_switch'
        method_gd: gradient-based optimisation method set to eithr 'SGD' or 'NSGD'
    """
    
    def __init__(self, X, B, m, s, p, l, model, model_name = 'gaussian', method_gd = 'SGD'):
        self.model = model
        self.B = B  
        self.m = m  
        self.s = s  
        self.p = p  
        self.X = X  
        self.n, self.d = self.X.shape   # dims of data
        self.method_gd = method_gd 
        self.model_name  = model_name 
        self.l = l  
        # median heuristic if l=-1 
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
            wll_j = 1 # dummy
            was_j = 1 # dummy for now
        
        # compute MMD minimizer
        #theta_j, loss_j = self.minimise_MMD(self.X, weights) 
        theta_j = self.minimise_MMD_scipy(self.X, weights)  
        loss_j = 1 # dummy
        # compute loss 
#        if self.model_name == 'gandk':
#            sample, z = self.model.sample(theta_j)
#        else:
#            sample = self.model.sample(theta_j)
#          
#        kyy = k(sample,sample,self.l)[0]
#        kxy = k(sample,self.X,self.l)[0]  
#        loss_j = self.MMD_approx(1/self.n*np.ones(self.n),kxy,kyy)
        
        return theta_j, wll_j, was_j, loss_j
        
    
    def draw_samples(self):
        """Draws B samples in parallel from the nonparametric posterior"""
        
        sample = np.zeros((self.B,self.p))
        wll = np.zeros((self.B,self.p))
        mmd_loss = np.zeros((self.B))
        was_sample = np.zeros((self.B,self.p))
        
        # Parallelize
        temp = Parallel(n_jobs=-1, backend='multiprocessing', max_nbytes=None,batch_size="auto")(delayed(self.draw_single_sample)(i) for i in tqdm(range(self.B)))
    
        for i in range(self.B):
            sample[i,:] = temp[i][0]
            wll[i,:] = temp[i][1]
            was_sample[i,:] = temp[i][2]
            mmd_loss[i] = temp[i][3]
           
                
        self.sample = np.array(sample)
        self.wll = np.array(wll)
        self.was = np.array(was_sample)
        self.mmd_loss = np.array(mmd_loss)
    
    
    def g_approx(self,grad_g, k2yy):
        """Approximation of the information metric """
        
        grad_g_T = np.einsum('ijk -> jik',grad_g)
        prod1 = np.einsum('ijk, jlkm -> ilkm', grad_g_T, k2yy)
        prod2 = np.einsum('ijkl,jmk->imkl', prod1, grad_g)
        for i in range(self.p):
            np.fill_diagonal(prod2[i,i,:,:], 0)
        gsum = np.einsum('ijkl->ij', prod2)
    
        return 1/(self.m*(self.m-1))*gsum

    
    def MMD_approx(self,weights,kxy,kyy):
        """ Approximates the MMD between P_\theta with y_{i=1}^m iid sample from P_\theta and
        P^j = \sum_{i=1}^n w^j_i \delta_{x_i}
        """
        
        # first sum
        np.fill_diagonal(kyy, np.repeat(0,self.m)) # excludes k(y_i, y_i) (diagonal terms)
        sum1 = np.sum(kyy)
    
        # second sum
        sum2 = np.sum(kxy*weights)
    
        # third sum
        np.fill_diagonal(self.kxx, np.repeat(0,self.n))
        kxx = self.kxx*weights # auto einai lathos!!! thelei dipla weights
        sum3 = np.sum(kxx)
    
        return (1/(self.m*(self.m-1)))*sum1-(2/(self.m))*sum2+(1/(self.n-1))*sum3
    
    
    def grad_MMD(self,weights,grad_g,k1yy,k1xy):
        """Approximates the gradient of the MMD with respect to \theta"""
        
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
        for i in range(self.d):
            k1xy[i,:,:] = k1xy[i,:,:]*weights
        # 
        prod2 = np.squeeze(np.einsum('ilj,imjk->lmjk', grad_g, np.expand_dims(k1xy,axis=1)))
        if prod2.ndim==2:
            sum2 = np.sum(prod2)
        else:
            sum2 = np.einsum('ijk->i',prod2)
    
        return (2/(self.m*(self.m-1)))*sum1-(2/(self.m))*sum2


    def minimise_MMD(self, data, weights, Nstep=10000, eta=0.1):
        """ Gradient-based optimization to minimize the MMD objective between 
        P^(j) for alpha=0 and P_\theta with
        Minibatch size = sample size = n, number of iterations = Nstep = 1000 and 
        step size = eta = 0.1"""
        # So far this is for gaussian or gandk 
        
        theta = 0.5*np.ones(self.p)
        t = 0
        gradient = 100
        #current_loss = 10
        while t < Nstep: 
            
            if self.model_name == 'gandk':
                sample, z = self.model.sample(theta)
                grad_g = self.model.grad_generator(z, theta)
            else:
                sample = self.model.sample(theta)
                grad_g = self.model.grad_generator(theta)
            kyy, k1yy, k2yy = k(sample,sample,self.l)
            kxy, k1xy = k(sample,data,self.l)[0:2]    #this is actually kyx - change name to avoid confusion
            
            # approximate MMD gradient
            if self.p == 1:
                gradient = np.asmatrix(self.grad_MMD(weights,grad_g,k1yy,k1xy))
            else:
                gradient = self.grad_MMD(weights,grad_g,k1yy,k1xy)
                
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
        
        loss = self.MMD_approx(weights,kxy,kyy)
        return theta, loss
    
    def minimise_MMD_scipy(self, data, weights):
        def MMD(theta):
            #np.random.seed(11)   # set seed so that MMD and mmd_grad use the same sample (for the same theta) at each iteration of the optimisation step
            if self.model_name == 'gandk':
                sample, z = self.model.sample(theta)
            else:
                sample = self.model.sample(theta)
          
            kyy, k1yy, k2yy = k(sample,sample,self.l)
            kxy, k1xy = k(sample,data,self.l)[0:2]    
        
            # first sum
            np.fill_diagonal(kyy, np.repeat(0,self.m)) # exclude k(y_i, y_i) (diagonal terms)
            sum1 = np.sum(kyy)
    
            # second sum
            sum2 = np.sum(kxy*weights)
    
            # third sum
            np.fill_diagonal(self.kxx, np.repeat(0,self.n))
            sum3 = np.sum(self.kxx*weights)
    

            return (1/(self.m*(self.m-1)))*sum1-(2/(self.m))*sum2+(1/(self.n-1))*sum3
    
        def mmd_grad(theta): # again so far I don't have grad for toggle switch
            np.random.seed(11)
            if self.model_name == 'gandk':
                    sample, z = self.model.sample(theta)
                    grad_g = self.model.grad_generator(z, theta)
            else:
                sample = self.model.sample(theta)
                grad_g = self.model.grad_generator(theta)
            
            kyy, k1yy, k2yy = k(sample,sample,self.l)
            kxy, k1xy = k(sample,data,self.l)[0:2]    #kyx
            
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
            for i in range(self.d):
                k1xy[i,:,:] = k1xy[i,:,:]*weights
            
            prod2 = np.squeeze(np.einsum('ilj,imjk->lmjk', grad_g, np.expand_dims(k1xy,axis=1)))
            if prod2.ndim==2:
                sum2 = np.sum(prod2)
            else:
                sum2 = np.einsum('ijk->i',prod2)
           
            return (2/(self.m*(self.m-1)))*sum1-(2/(self.m))*sum2
        
        if self.model_name == 'gaussian':
            x0 = np.mean(self.X, axis=0) # mle
        elif self.model_name == 'gandk':
            x0 = np.zeros(self.p)
            x0[0] = np.median(self.X)
            x0[1] = stats.iqr(self.X, interpolation = 'midpoint')/2
            x0[2] = 0
            x0[3] = 0
        elif self.model_name == 'toggle_switch':
            x0 = np.ones(self.p)
            x0[0] = 20
            x0[1] = 10
            x0[2] = 2
            x0[3] = 2
            x0[4] = 250
            x0[5] = 0
            x0[6] = 0
        
        if self.model_name == 'toggle_switch':
            #grad_mmd = grad(MMD)
            optimization_result = minimize(MMD, x0 , method= 'Nelder-Mead', options={'disp': True})
        else:
            optimization_result = minimize(MMD, x0 ,  method= 'BFGS', options={'disp': False})
        # bug: mmd_grad arg doesn't work for d=1 --- need to fix #jac=mmd_grad,
        
        # return the value at optimum
        return optimization_result.x
        
    def WLL(self, data, weights):
        """Get weighted log likelihood minimizer, for gaussian model""" 
        theta = 0
        for i in range(self.n):
            theta += weights[i]*data[i] 
        return theta
    
    def minimise_wasserstein(self, data, weights):
        def wasserstein(theta):
            a = np.ones((self.m,)) / self.m 
            b = weights #np.ones((self.n,)) / self.n
            sample = self.model.sample(theta)
         
            M = ot.dist(sample, data, 'euclidean')

            W = ot.emd2(a, b, M)
            return W
        
        optimization_result = minimize(wasserstein, np.mean(data,axis=0), method= 'Powell', options={'disp': False, 'maxiter': 10000})
     
        # return the value at optimum
        return optimization_result.x
    
