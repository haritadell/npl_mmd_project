#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 18:55:21 2021

@author: HaritaDellaporta
"""

from utils import sample_gandk_outl
from plot_functions import plot_posterior_marginals, plot_mse, plot_gnk
import NPL
import models
import numpy as np
import pandas as pd
from scipy import stats

# Set paths
data_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/G_and_k_model/"
results_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/G_and_k_model/"
plots_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/Plots/G_and_k/"

# Set to True to generate and save fresh datasets or False to load saved datasets
sample_data_bool = False

# Set model 
model_name = 'gandk' 
n = 2**11
d = 1
theta_star = np.array([3,1,1,-np.log(2)]) 
outl = 3 # number of different percentages of outliers to run for
m = 2**9 # number of samples
l = 0.15  # kernel lengthscale
p = 4   # number of unknown parameters
B = 512 # number of bootstrap iterations 
model = models.g_and_k_model(m,d)
R = 10 # number of independent runs
s=1 # std of Gaussian data

## Sample R sets of data
if sample_data_bool:
    for j in range(R):
        for i in range(outl):
          X = sample_gandk_outl(n,d,s,theta_star, n_cont=i)
          np.savetxt(data_path+'run_{}_outl_{}'.format(j,i), X)

# Load data
datasets = np.zeros((R,outl,n))
for j in range(R):
    for i in range(outl):
        X = np.loadtxt(data_path+'run_{}_outl_{}'.format(j,i))
        datasets[j,i,n] = X
    
# Obtain and save results 
summary_stats = np.zeros((R,outl, p, 4)) # collect mean, median, mode, st.dev for each bootstrap sample
for j in range(R):
    print("-----Run ", j)
    for n_cont in range(outl):
        print("-----Running for", n_cont*5, "% of outliers-----")
        X =datasets[j,n_cont,:].reshape((n,1))
        npl = NPL.npl(X,B,m,p,l, model = model, model_name = model_name)
        npl.draw_samples()
        sample = npl.sample
        summary_stats[j,n_cont,:,0] = np.mean(sample, axis=0)
        summary_stats[j,n_cont,:,1] = np.std(sample, axis=0)
        summary_stats[j,n_cont,:,2] = np.median(sample, axis=0)
        summary_stats[j,n_cont,:,3] = stats.mode(sample, axis=0)[0]
        np.savetxt(results_path+'NPL_MMD/thetas_mmd_outl_{}_run_{}.txt'.format(n_cont,j), sample)
        
np.savetxt(results_path+'NPL_MMD/summary_stats.txt', summary_stats)
      

#### Reshaping and plotting results for a single run   
r = 0 # index which run you want results for
# Reshape results
thetas_mmd = np.zeros((p,outl,B))
for i in range(outl):
    for j in range(p):
        sample = np.loadtxt(results_path+'NPL_MMD/thetas_mmd_outl_{}_run_{}.txt'.format(i,r))
        thetas_mmd[j,i,:] = sample[:,j]

names = ["A", "B", "g", "k"]
thetas_wabc = np.zeros((p,outl,B))
for n_cont in range(outl):
    for i,name in enumerate(names):
        df = pd.read_csv(results_path+'WABC/thetas_wabc_outl_{}_run_{}.csv'.format(n_cont,r))
        thetas_wabc[i,n_cont,:] = df[name]
thetas_wabc[3,:,:] = np.log(thetas_wabc[3,:,:])   # last parameter for gandk is exp(k)
      
# Plot results
fname1 = plots_path+'run_{}_post_marg'.format(r)
fname2 = plots_path+'run_{}_mse'.format(r)
#plot_posterior_marginals(B,thetas_wabc, thetas_mmd, thetas_mmd,thetas_mmd, theta_star, outl, save_fig=False)
plot_mse(thetas_wabc, thetas_mmd,thetas_mmd,thetas_mmd, thetas_mmd, theta_star, outl, fname2, save_fig=False)
plot_gnk(B,thetas_wabc, thetas_mmd, theta_star,outl, fname1, save_fig=False)