#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 19:05:26 2021

@author: HaritaDellaporta
"""

from utils import sample_togswitch_noise
from plot_functions import plot_posterior_marg_ts
import NPL
import models
import numpy as np
import pandas as pd
from scipy import stats

#
data_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/Toggle_switch_model/"
results_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/Toggle_switch_model/"
plots_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/Plots/Toggle_switch/" 

# Set to True to generate and save fresh datasets or False to load saved datasets
sample_data_bool = False

# Set model 
model_name = 'toggle_switch' 
n = 2000
d = 1
theta_star = np.array([22.,np.log(12),4.,4.5,325.,0.25,0.15])
m = 500 # number of samples
p = 7   # number of unknown parameters
B = 300 # number of bootstrap iterations 
T = 300 
model = models.toggle_switch_model(m,d,T)
R = 10 # number of independent runs
df = 1 # degrees of freedom for random noise
l = 1

## Sample or load R sets of data
if sample_data_bool:
    for j in range(R):
        X = sample_togswitch_noise(theta_star,n,T,df,add_noise=True)
        np.savetxt(data_path+'run_{}'.format(j), X)

# Load data
datasets = np.zeros((R,n))
for j in range(R):
    X = np.loadtxt(data_path+'run_{}'.format(j))
    datasets[j,n] = X
    
# Obtain and save results 
summary_stats = np.zeros((R, p, 4)) # collect mean, median, mode, st.dev for each bootstrap sample
for j in range(R):
    print("-----Run ", j)
    X =datasets[j,:].reshape((n,1))
    npl = NPL.npl(X,B,m,p,l, model = model, model_name = model_name)
    npl.draw_samples()
    sample = npl.sample
    #wasserstein_sample = npl.was
    summary_stats[j,:,0] = np.mean(sample, axis=0)
    summary_stats[j,:,1] = np.std(sample, axis=0)
    summary_stats[j,:,2] = np.median(sample, axis=0)
    summary_stats[j,:,3] = stats.mode(sample, axis=0)[0]
    np.savetxt(results_path+'NPL_MMD/thetas_mmd_run_{}.txt'.format(j), sample)
    
np.savetxt(results_path+'NPL_MMD/summary_stats.txt', summary_stats)
      

#### Reshaping and plotting results for a single run   
r = 0 # index which run you want results for
# Reshape results
thetas_mmd = np.zeros((p,B))
for j in range(p):
    sample = np.loadtxt(results_path+'NPL_MMD/thetas_mmd_run_{}.txt'.format(r))
    if p>1:
        thetas_mmd[j,:] = sample[:,j]
    else:
        thetas_mmd[j,:] = sample
            
names = ["alpha_1", "alpha_2", "beta_1", "beta_2", "mu", "sigma", "gamma"]
thetas_wabc = np.zeros((p,B))
for i,name in enumerate(names):
    df = pd.read_csv(results_path+'WABC/thetas_wabc_outl_{}_run_{}.csv'.format(i,r))
    thetas_wabc[i,:] = df[name]
thetas_wabc[2,:] = np.log(thetas_wabc[2,:])
      
# Plot results
fname = plots_path+'run_{}_post_marg'.format(r)
plot_posterior_marg_ts(B,thetas_wabc, thetas_mmd, theta_star, fname, save_fig=False)