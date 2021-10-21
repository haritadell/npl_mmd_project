#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 19:05:26 2021

@author: HaritaDellaporta
"""

from utils import sample_togswitch_noise
from plot_functions import plot_posterior_marg_tsols
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
theta_star = np.array([22.,12.,4.,4.5,325.,0.25,0.15])
m = 500 # number of samples
p = 7   # number of unknown parameters
B = 300 # number of bootstrap iterations 
T = 300 
model = models.toggle_switch_model(m,d,T)
R = 5 # number of independent runs
df = 1 # degrees of freedom for random noise
l = 1

## Sample or load R sets of data
if sample_data_bool:
    for j in range(R):
        X, noise, y = sample_togswitch_noise(theta_star,n,T,df,add_noise=True)
        np.savetxt(data_path+'run_{}'.format(j), X)
        if j == 0:
            np.savetxt(data_path+'run_{}_y'.format(j), y)
            np.savetxt(data_path+'run_{}_noise'.format(j), noise)
#%%
# Load data
datasets = np.zeros((R,n))
for j in range(R):
    X = np.loadtxt(data_path+'run_{}'.format(j))
    datasets[j,:] = X
times = []
# Obtain and save results 
summary_stats = np.zeros((R, p, 4)) # collect mean, median, mode, st.dev for each bootstrap sample
for j in range(R):
    print("-----Run ", j)
    X =datasets[j,:].reshape((n,1))
    npl = NPL.npl(X,B,m,p,l, model = model, model_name = model_name)
    t0 = time.time()
    npl.draw_samples()
    t1 = time.time()
    total = t1-t0
    times.append(total)
    print(total)
    sample = npl.sample
    summary_stats[j,:,0] = np.mean(sample, axis=0)
    summary_stats[j,:,1] = np.std(sample, axis=0)
    summary_stats[j,:,2] = np.median(sample, axis=0)
    summary_stats[j,:,3] = stats.mode(sample, axis=0)[0]
    np.savetxt(results_path+'NPL_MMD/thetas_mmd_run_{}.txt'.format(j), sample)
    
np.savetxt(results_path+'NPL_MMD/summary_stats.txt', summary_stats)
np.savetxt(results_path+'NPL_MMD/cpu_times.txt', times)      
#%%
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
thetas_mmd[0,:] = np.exp(thetas_mmd[0,:]) 
thetas_mmd[1,:] = np.exp(thetas_mmd[1,:]) 
thetas_mmd[4,:] = np.exp(thetas_mmd[4,:])  
thetas_mmd[5,:] = np.exp(thetas_mmd[5,:]) 
          
names = ["alpha_1", "alpha_2", "beta_1", "beta_2", "mu", "sigma", "gamma"]
thetas_wabc = np.zeros((p,B))
for i,name in enumerate(names):
    df = pd.read_csv(results_path+'WABC/thetas_wabc_run_{}.csv'.format(r))
    thetas_wabc[i,:] = df[name]
     
# Plot results
y = np.loadtxt(data_path+'run_0_y')
fname = plots_path+'run_{}_post_marg_resized'.format(r)
plot_posterior_marg_tsols(B,thetas_wabc, thetas_mmd, theta_star, y,fname, save_fig=True)