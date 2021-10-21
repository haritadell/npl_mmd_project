#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:36:07 2021

@author: HaritaDellaporta
"""

from utils import sample_gaussian_outl
from plot_functions import plot_posterior_marginals, plot_mse, SeabornFig2Grid
import NPL
import models
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.gridspec as gridspec

#
data_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/Gaussian_location_model/"
results_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/Gaussian_location_model/"
plots_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/Plots/"

# Set to True to generate and save fresh datasets or False to load saved datasets
sample_data_bool = False

# Set model 
model_name = 'gaussian' 
n = 200
d = 4
theta_star = np.array([1,1,np.exp(1), np.exp(1)])#np.ones(d) 
s = 1 # std for gaussian model
outl = 3 # number of different percentages of outliers to run for
m = 200 # number of samples
l = -1  # kernel lengthscale
p = d  # number of unknown parameters
B = 500 # number of bootstrap iterations 
model = models.gauss_model(m,d,s)
R = 10 # number of independent runs
#%%
# Sample and save R sets of data
if sample_data_bool:
    for j in range(R):
        for i in range(outl):
            X = sample_gaussian_outl(n,d,s,theta_star, n_cont=i)
            np.savetxt(data_path+'run_{}_outl_{}_dim_{}'.format(j,i,d), X)

# Load sata
datasets = np.zeros((R,outl,n,d))
for j in range(R):
    for i in range(outl):
        X = np.loadtxt(data_path+'run_{}_outl_{}_dim_{}'.format(j,i,d))
        datasets[j,i,:,:] = X
#%%   
# Obtain and save results
summary_stats = np.zeros((R,outl, p, 4)) # collect mean, median, mode, st.dev for each bootstrap sample
summary_stats_wll = np.zeros((R,outl, p, 4))
times = []
for j in range(R):
    print("-----Run ", j)
    for n_cont in range(outl):
        print("-----Running for", n_cont*5, "% of outliers-----")
        X =datasets[j,n_cont,:,:].reshape((n,d))
        npl = NPL.npl(X,B,m,p,l, model = model, model_name = model_name)
        t0 = time.time()
        npl.draw_samples()
        t1 = time.time()
        total = t1-t0
        times.append(total)
        print(total)
        sample = npl.sample
        wll_sample = npl.wll_sample
        wasserstein_sample = npl.was
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

np.savetxt(results_path+'NPL_MMD/cpu_times.txt', times)     
#%%
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
thetas_mmd[2,:,:] = np.exp(thetas_mmd[2,:,:])
thetas_mmd[3,:,:] = np.exp(thetas_mmd[3,:,:])
thetas_wll[2,:,:] = np.exp(thetas_wll[2,:,:])
thetas_wll[3,:,:] = np.exp(thetas_wll[3,:,:])
          
names = ["X1", "X2", "X3", "X4"]
thetas_wabc = np.zeros((p,outl,B))
for n_cont in range(outl):
    for i,name in enumerate(names):
        df = pd.read_csv(results_path+'WABC/thetas_wabc_outl_{}_run_{}_dim_{}.csv'.format(n_cont,r,d))
        thetas_wabc[i,n_cont,:] = df[name]
thetas_wabc[2,:,:] = np.exp(thetas_wabc[2,:,:])
thetas_wabc[3,:,:] = np.exp(thetas_wabc[3,:,:])

names = ["X1", "X2", "X3", "X4"]
thetas_mabc = np.zeros((p,outl,B))
for n_cont in range(outl):
    for i,name in enumerate(names):
        df = pd.read_csv(results_path+'WABC/thetas_mabc_outl_{}_run_{}_dim_{}.csv'.format(n_cont,r,d))
        df = df[0:500]
        thetas_mabc[i,n_cont,:] = df[name]
thetas_mabc[2,:,:] = np.exp(thetas_mabc[2,:,:])
thetas_mabc[3,:,:] = np.exp(thetas_mabc[3,:,:])
      
# Plot results
# Set names for save figures
fname1 = plots_path+'gauss_dim_{}_run_{}_post_marg'.format(d,r)
fname2 = plots_path+'gauss_dim_{}_run_{}_mse'.format(d,r) 

fig = plt.figure(figsize=(13,5))
gs = gridspec.GridSpec(2, 3)
fname1 = plots_path+'run_{}_post_marg_resized'.format(r)
#axes = plot_gauss_4d(B,thetas_wabc, thetas_mmd, thetas_wll, thetas_wll, theta_star,outl, fname1,save_fig=False)
axes = plot_gauss_4d_2(B, thetas_mmd, thetas_mabc, theta_star, outl, fname1)
mg0 = SeabornFig2Grid(axes[0], fig, gs[0], '$\epsilon = 0$', add_title=True)
mg1 = SeabornFig2Grid(axes[1], fig, gs[1], '$\epsilon = 0.05$', add_title=True)
mg2 = SeabornFig2Grid(axes[2], fig, gs[2], '$\epsilon = 0.1$', add_title=True)
mg3 = SeabornFig2Grid(axes[3], fig, gs[3], '0%')
mg4 = SeabornFig2Grid(axes[4], fig, gs[4], '0%')
mg5 = SeabornFig2Grid(axes[5], fig, gs[5], '0%')

gs.tight_layout(fig)
#plt.savefig(fname1)
plt.show()

