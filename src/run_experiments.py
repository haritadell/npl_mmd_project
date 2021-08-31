#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:02:42 2021

"""

from utils import sample_gaussian_outl, sample_gandk_outl, sample_togswitch_outl2
from plot_functions import plot_posterior_marginals, plot_mse, plot_gnk
import NPL
import models
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd


#%%
# Set model 
model_name = 'gandk' # set to gaussian or gandk or toggle_switch

# Sample and save data
np.random.seed(11)
n = 2048
d = 1
theta_star = np.array([3,1,1,-np.log(2)])     #np.ones(d) #np.array([22,12,4,4.5,325,0.25,0.15]) #np.array([3,1,1,-np.log(2)]) 
s = 1 # std for gaussian model
outl = 3 # number of different percentages of outliers to run for
T = 20 # for toggle switch model
#%%
for i in range(outl):
    if model_name == 'gaussian':
        X = sample_gaussian_outl(n,d,s,theta_star, n_cont=i)
        #X = np.random.gamma(10,0.5,n)   # misspecification case
    elif model_name == 'gandk':
        X = sample_gandk_outl(n,d,theta_star, n_cont=i)
    else:
        X = sample_togswitch_outl2(theta_star, n, T, n_cont=i)  # ftiaxe to fn twra pou to eluses to thema
    np.savetxt("/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/data_{}_{}_{}_{}.txt".format(i,n,model_name,d),X)
#%%    
# Set parameters
m = 2**9 # number of samples
l = 0.1  # kernel lengthscale
p = 4 #4,7,2   # number of unknown parameters
B = 512 # number of bootstrap iterations 
#%%
method = 'SGD'  # Optimisation method 
if model_name == 'gaussian':
    model = models.gauss_model(m,d,s)   # set model
elif model_name == 'gandk':
    model = models.g_and_k_model(m,d)
else:
    model = models.toggle_switch_model(m,d,T)
    
# Optimisation functions for NPL MMD

def optim_gaus(X1,X2,X3,n,m,s,l,theta_star,d,p,B,outl,method,model_name,model): 
    results = np.zeros((3,p,outl)) # each of the rows are MMD, MLE, WLL resp.
    samples = np.zeros((outl,B,p))
    wll_samples = np.zeros((outl,B,p))
    wasserstein_samples = np.zeros((outl,B,p))

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
        wll_samples[n_cont,:,:] = wll_sample
        wasserstein_sample = npl.was
        wasserstein_samples[n_cont,:,:] = wasserstein_sample
        np.savetxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/outl_{}_{}_{}_gaussian_missp2.txt'.format(n_cont,n,d), sample)
        np.savetxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/outl_{}_{}_{}_gaussian_wll_missp2.txt'.format(n_cont,n,d), wll_sample)
        np.savetxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/outl_{}_{}_{}_gaussian_was_missp2.txt'.format(n_cont,n,d), wasserstein_sample)
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
        if n_cont == 0:
            X = X2    # allaxe
        elif n_cont == 1:
            X = X3
        elif n_cont == 2:
            X = X2
    
        npl = NPL.npl(X,B,m,s,p,l, model = model, model_name = model_name, method_gd = method)
        npl.draw_samples()
        sample = npl.sample
        samples[n_cont,:,:] = sample
        np.savetxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/outl_{}_{}_gandk_.txt'.format(n_cont+1,n), sample)
   
        for i in range(p):
            results[i,n_cont] = mean_squared_error(theta_star[i]*np.ones(B), sample[:,i])
        
    return results, samples

def optim_togswitch(X1,X2,X3,n,m,T,l,theta_star,d,p,B,outl,method,model_name,model):
    results = np.zeros((p,outl))
    samples = np.zeros((outl,B,p))
    for n_cont in range(outl):
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
        np.savetxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/outl_{}_{}_togswitch.txt'.format(n_cont,n), sample)
   
        for i in range(p):
            results[i,n_cont] = mean_squared_error(theta_star[i]*np.ones(B), sample[:,i])
        
    return results, samples

#%%
# load data 
X1 = np.reshape(np.loadtxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/data_0_{}_{}_{}.txt'.format(n,model_name,d)), (n,d))
X2 = np.reshape(np.loadtxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/data_1_{}_{}_{}.txt'.format(n,model_name,d)), (n,d))
X3 = np.reshape(np.loadtxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/data_2_{}_{}_{}.txt'.format(n,model_name,d)), (n,d))

# run 
if model_name == 'gandk':
    results, samples = optim_gandk(X1,X2,X3,n,m,s,l,theta_star,d,p,B,outl,method,model_name,model) 
elif model_name == 'gaussian':
    results, samples = optim_gaus(X1,X1,X1,n,m,s,l,theta_star,d,p,B,outl,method,model_name,model)
else:
    results, samples = optim_togswitch(X1,X1,X1,n,m,T,l,theta_star,d,p,B,outl,method,model_name,model)
 #%%
# Reshape results
outl=3
thetas_mmd = np.zeros((p,outl,B))
thetas_wll = np.zeros((p,outl,B))
thetas_was = np.zeros((p,outl,B))
for n_cont in range(outl):
    for j in range(p):
        #sample = np.loadtxt('/Users/HaritaDellaporta/Dropbox/sample_{}_gk_2048.txt'.format(n_cont))
       # if n_cont == 2:
            #sample = np.loadtxt('/Users/HaritaDellaporta/Dropbox/thetas_mmd_{}_gauss_{}_{}.txt'.format(n_cont,n,d))
        #else:
        #sample = np.loadtxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/Gaussian_location_model/NPL_MMD/thetas_mmd_{}_gauss_{}_{}.txt'.format(n_cont,n,d))
        sample = np.loadtxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/G_and_k_model/NPL_MMD/sample_{}_gk_{}_new_.txt'.format(n_cont,n)) #
        #sample_wll = np.loadtxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/Gaussian_location_model/NPL_WLL/thetas_wll_{}_gauss_{}_{}.txt'.format(n_cont,n,d))
        #sample_was = np.loadtxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/outl_{}_{}_{}_{}_was_missp.txt'.format(n_cont,n,d,model_name))
        if p>1:
            thetas_mmd[j,n_cont,:] = sample[:,j]
            #thetas_wll[j,n_cont,:] = sample_wll[:,j]
            #thetas_was[j,n_cont,:] = sample_was[:,j]
        else:
            thetas_mmd[j,n_cont,:] = sample
            #thetas_wll[j,n_cont,:] = sample_wll
            #thetas_was[j,n_cont,:] = sample_was
            
#%%

#%%    
# Import data from W-ABC method
   
if model_name == 'gandk':
    names = ["A", "B", "g", "k"]
    thetas_wabc = np.zeros((p,outl,B))
    for n_cont in range(outl):
        for i,name in enumerate(names):
            df = pd.read_csv('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results//G_and_k_model/WABC/results_{}_gandk_{}_new.csv'.format(n_cont,n))
            thetas_wabc[i,n_cont,:] = df[name]
    thetas_wabc[3,:,:] = np.log(thetas_wabc[3,:,:])   # last parameter for gandk is exp(k)
elif model_name == 'gaussian':
    names = ["X1", "X2"]
    thetas_wabc = np.zeros((p,outl,B))
    for n_cont in range(outl):
        for i,name in enumerate(names):
            df = pd.read_csv('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/Gaussian_location_model/WABC/results_{}_{}_gauss_{}.csv'.format(n,d,n_cont))
            thetas_wabc[i,n_cont,:] = df[name]
elif model_name == 'toggle_switch':
    names = ["alpha_1", "alpha_2", "beta_1", "beta_2", "mu", "sigma", "gamma"]
    thetas_wabc = np.zeros((p,outl,B))
    for n_cont in range(outl):
        for i,name in enumerate(names):
            df = pd.read_csv('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/results_togswitch_2000.csv'.format(B))
            df = df[0:100]
            thetas_wabc[i,n_cont,:] = df[name]
    thetas_wabc[1,:,:] = np.log(thetas_wabc[1,:,:])
    thetas_wabc[5,:,:] = np.log(thetas_wabc[5,:,:])
    

#%%          
# Plots
if model_name == 'gaussian':
    # Calculate mse for mle
    mse_MLE = np.zeros((outl,p))
    mle1 = np.mean(X1, axis=0)
    mse_MLE[0,:] = (mle1-theta_star)**2
    mle2 = np.mean(X2, axis=0)
    mse_MLE[1,:] = (mle2-theta_star)**2
    mle3 = np.mean(X3, axis=0)
    mse_MLE[2,:] = (mle3-theta_star)**2
    #plots
    plot_posterior_marginals(B,thetas_wabc, thetas_mmd, thetas_wll, thetas_mmd, theta_star, outl, gaussian=True, save_fig=True)
    plot_mse(thetas_wabc, thetas_mmd, thetas_wll, thetas_was, np.zeros((outl,p)), theta_star, outl, gaussian=False, save_fig=True)
elif model_name == 'gandk':
    plot_posterior_marginals(B,thetas_wabc, thetas_mmd, thetas_mmd,thetas_mmd, theta_star, outl, save_fig=False)
    plot_mse(thetas_wabc, thetas_mmd,thetas_mmd,thetas_mmd, thetas_mmd, theta_star, outl, save_fig=False)
    plot_gnk(B,thetas_wabc, thetas_mmd, theta_star,outl, save_fig=False)
elif model_name == 'toggle_switch':
    plot_posterior_marginals(B,thetas_wabc, thetas_mmd, thetas_mmd,thetas_mmd, theta_star, outl, save_fig=False)
    plot_mse(thetas_wabc, thetas_mmd,thetas_mmd,thetas_mmd, thetas_mmd, theta_star, outl, save_fig=False)
    
    




