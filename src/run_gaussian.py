#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:36:07 2021

@author: HaritaDellaporta
"""

from utils import sample_gaussian_outl
from plot_functions import plot_posterior_marginals, plot_mse
import NPL
import models
import numpy as np
import pandas as pd
from scipy import stats

#
data_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/Gaussian_location_model/"
results_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/Gaussian_location_model/"
plots_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/Plots/"

# Set to True to generate and save fresh datasets or False to load saved datasets
sample_data_bool = False

# Set model 
model_name = 'gaussian' 
n = 2000
d = 1
theta_star = np.ones(d) 
s = 1 # std for gaussian model
outl = 3 # number of different percentages of outliers to run for
m = 500 # number of samples
l = -1  # kernel lengthscale
p = d  # number of unknown parameters
B = 512 # number of bootstrap iterations 
model = models.gauss_model(m,d,s)
R = 10 # number of independent runs

# Sample and save R sets of data
if sample_data_bool:
    for j in range(R):
        for i in range(outl):
            X = sample_gaussian_outl(n,d,s,theta_star, n_cont=i)
            np.savetxt(data_path+'run_{}_outl_{}_dim_{}'.format(j,i,d), X)

# Load sata
datasets = np.zeros((R,outl,n))
for j in range(R):
    for i in range(outl):
        X = np.loadtxt(data_path+'run_{}_outl_{}_dim_{}'.format(j,i,d))
        datasets[j,i,n] = X
    
# Obtain and save results
summary_stats = np.zeros((R,outl, p, 4)) # collect mean, median, mode, st.dev for each bootstrap sample
summary_stats_wll = np.zeros((R,outl, p, 4))
for j in range(R):
    print("-----Run ", j)
    for n_cont in range(outl):
        print("-----Running for", n_cont*5, "% of outliers-----")
        X =datasets[j,n_cont,:].reshape((n,1))
        npl = NPL.npl(X,B,m,p,l, model = model, model_name = model_name)
        npl.draw_samples()
        sample = npl.sample
        wll_sample = npl.wll
        #wasserstein_sample = npl.was
        summary_stats[j,n_cont,:,0] = np.mean(sample, axis=0)
        summary_stats_wll[j,n_cont,:,0] = np.mean(wll_sample, axis=0)
        summary_stats[j,n_cont,:,1] = np.std(sample, axis=0)
        summary_stats_wll[j, n_cont, :, 1] = np.std(wll_sample, axis=0)
        summary_stats[j,n_cont,:,2] = np.median(sample, axis=0)
        summary_stats_wll[j,n_cont,:,2] = np.median(wll_sample, axis=0)
        summary_stats[j,n_cont,:,3] = stats.mode(sample, axis=0)[0]
        summary_stats_wll[j,n_cont,:,3] =  stats.mode(wll_sample, axis=0)[0]
        np.savetxt(results_path+'NPL_MMD/thetas_mmd_outl_{}_run_{}_dim_{}.txt'.format(n_cont,j,d), sample)
        np.savetxt(results_path+'NPL_WLL/thetas_wll_outl_{}_run_{}_dim_{}.txt'.format(n_cont,j,d), sample)
        
np.savetxt(results_path+'NPL_MMD/summary_stats.txt', summary_stats)
np.savetxt(results_path+'NPL_WLL/summary_stats.txt', summary_stats_wll)
      

#### Reshaping and plotting results for a single run   
r = 0 # index which run you want results for
# Reshape results
thetas_mmd = np.zeros((p,outl,B))
thetas_wll = np.zeros((p,outl,B))
#thetas_was = np.zeros((p,outl,B))
for i in range(outl):
    for j in range(p):
        sample = np.loadtxt(results_path+'NPL_MMD/thetas_mmd_outl_{}_run_{}_dim_{}.txt'.format(i,r,d))
        sample_wll = np.loadtxt(results_path+'NPL_WLL/thetas_wll_outl_{}_run_{}_dim_{}.txt'.format(i,r,d))
        #sample_was = np.loadtxt('/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/outl_{}_{}_{}_{}_was_missp.txt'.format(n_cont,n,d,model_name))
        if p>1:
            thetas_mmd[j,i,:] = sample[:,j]
            thetas_wll[j,i,:] = sample_wll[:,j]
            #thetas_was[j,n_cont,:] = sample_was[:,j]
        else:
            thetas_mmd[j,i,:] = sample
            thetas_wll[j,i,:] = sample_wll
            #thetas_was[j,n_cont,:] = sample_was
            
names = ["X1", "X2"]
thetas_wabc = np.zeros((p,outl,B))
for n_cont in range(outl):
    for i,name in enumerate(names):
        df = pd.read_csv(results_path+'WABC/thetas_wabc_outl_{}_run_{}_dim_{}.csv'.format(n_cont,r,d))
        thetas_wabc[i,n_cont,:] = df[name]
      
# Plot results
# Set names for save figures
fname1 = plots_path+'gauss_dim_{}_run_{}_post_marg'.format(d,r)
fname2 = plots_path+'gauss_dim_{}_run_{}_mse'.format(d,r) 
plot_posterior_marginals(B,thetas_wabc, thetas_mmd, thetas_wll, thetas_mmd, theta_star, outl, fname1, gaussian=True, save_fig=False)
plot_mse(thetas_wabc, thetas_mmd, thetas_wll, thetas_wll, np.zeros((outl,p)), theta_star, outl, fname2, gaussian=True, save_fig=False)



