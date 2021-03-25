#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:02:42 2021

@author: HaritaDellaporta
"""

from utils import sample_gaussian_outl, sample_gandk_outl
from plot_functions import plot_posterior_marginals, plot_mse
import NPL
import models
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

# Set model 
model_name = 'gaussian' # set to gaussian or g_and_k 

# Sample and save data
np.random.seed(11)
n = 200
d = 2
theta_star = np.ones(d)
s = 1
outl = 3 # number of different percentages of outliers to run for
for i in range(outl):
    if model_name == 'gaussian':
        X = sample_gaussian_outl(n,d,s,theta_star, n_cont=i)
    elif model_name == 'gandk':
        X = sample_gandk_outl(n,d,theta_star, n_cont=i)
    np.savetxt("data_{}_{}.txt".format(i,n))
    
# Set parameters
m = 2**9 # number of samples
l = -1  # kernel lengthscale
p = 2   # number of unknown parameters
B = 512 # number of bootstrap iterations 
method = 'SGD'  # Optimisation method 
if model_name == 'gaussian':
    model = models.gauss_model(m,d,s)   # set model
elif model_name == 'gandk':
    model = models.gandk_model(m,d)
    
# Optimisation functions for NPL MMD

def optim_gaus(X1,X2,X3,n,m,s,l,theta_star,d,p,B,outl,method,model_name,model): 
    results = np.zeros((3,p,outl)) # each of the rows are MMD, MLE, WLL resp.
    samples = np.zeros((outl,B,p))

    for n_cont in np.arange(outl):
        print("-----Running for", n_cont*5, "% of outliers-----")
        if n_cont == 0:
            X = X1
        elif n_cont == 1:
            X = X2
        elif n_cont == 2:
            X = X3
        npl = NPL.npl(X,B,m,s,p,l, model = model, model_name = model_name, method_gd = method)
        npl.draw_samples()
        sample = npl.sample
        samples[n_cont,:,:] = sample
        wll_sample = npl.wll
        np.savetxt('outl_{}_{}_gaussian.txt'.format(n_cont,n), sample)
        sample_mle = np.mean(X, axis=0)
      
        for i in range(p):
            results[0, i, n_cont] = mean_squared_error(theta_star[i]*np.ones(B), sample[:,i])
            results[1, i, n_cont] = mean_squared_error([theta_star[i]], [sample_mle[i]])
            results[2, i, n_cont] = mean_squared_error(theta_star[i]*np.ones(B), wll_sample[:,i])  # need to store thetas as well 
        
    return results, samples
    
def optim_gandk(X1,X2,X3,n,m,s,l,theta_star,d,p,B,outl,method,model_name,model):
    results = np.zeros((p,outl))
    samples = np.zeros((outl,B,p))
    for n_cont in range(outl):
        print("-----Running for", n_cont*5, "% of outliers-----")
        if n_cont+1 == 0:
            X = X1
        elif n_cont+1 == 1:
            X = X2
        elif n_cont+1 == 2:
            X = X3
    
        npl = NPL.npl(X,B,m,s,p,l, model = model, model_name = model_name, method_gd = method)
        npl.draw_samples()
        sample = npl.sample
        samples[n_cont,:,:] = sample
        np.savetxt('outl_{}_{}_gandk.txt'.format(n_cont,n), sample)
   
        for i in range(p):
            results[i,n_cont] = mean_squared_error(theta_star[i]*np.ones(B), sample[:,i])
        
    return results, samples

# Reshape results 
thetas_mmd = np.zeros((p,outl,B))
for n_cont in range(outl):
    for j in range(p):
        sample = np.loadtxt('results/outl_{}_{}_{}.txt'.format(n_cont,n,model_name))
        thetas_mmd[j,n_cont,:] = sample[:,j]
        
# Import data from W-ABC method
        
if model_name == 'gandk':
    names = ["A", "B", "g", "k"]
    thetas_wabc = np.zeros((p,outl,B))
    for n_cont in range(outl):
        for i,name in enumerate(names):
            df = pd.read_csv('results/results_{}_gandk_{}.csv'.format(n,n_cont))
            thetas_wabc[i,n_cont,:] = df[name]
    thetas_wabc[3,:,:] = np.log(thetas_wabc[3,:,:])   # last parameter for gandk is exp(k)
elif model_name == 'gaussian':
    names = ["X1", "X2"]
    thetas_wabc = np.zeros((p,outl,B))
    for n_cont in range(outl):
        for i,name in enumerate(names):
            df = pd.read_csv('results/results_{}_gauss_{}.csv'.format(n,n_cont))
            thetas_wabc[i,n_cont,:] = df[name]
            
# Plots
plot_posterior_marginals(B,thetas_wabc, thetas_mmd, theta_star, outl)
plot_mse(thetas_wabc, thetas_mmd, theta_star, outl)
    