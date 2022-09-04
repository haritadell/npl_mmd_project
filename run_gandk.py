#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 18:55:21 2021

@author: HaritaDellaporta
"""
import sys
sys.path.append("./src")

from utils import sample_gandk_outl, k, MMD_approx
from plot_functions import plot_gnk, SeabornFig2Grid
import NPL
import NPL_prior
import models
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# Before running: 
# 1) Set paths 
# 2) Indicate whether you want a fresh dataset or to load existing one 
# 3) Experiments are run for multiple runs - index which run you want plots for

# Set paths
data_path = "./data/G_and_k_model/"
results_path = "./results/" #G_and_k_model/"
plots_path = "./plots/G_and_k/"

# Set to True to generate and save fresh datasets or False to load saved datasets
sample_data_bool = False

# Index which run you want to plot results for in [0,R-1]
r = 0 

# Set model 
model_name = 'gandk' 
n = 2**11
d = 1
theta_star = np.array([3,1,1,-np.log(2)]) 
outl = 3 # number of different percentages of outliers to run for
m = 2**9 # number of samples
l = 0.15  # kernel lengthscale
p = 4   # number of unknown parameters
B = 500 # number of bootstrap iterations 
model = models.g_and_k_model(m,d)
R = 10 # number of independent runs
s=1 # std of Gaussian data

## Sample R sets of data
if sample_data_bool:
    for j in range(R):
        for i in range(outl):
          X = sample_gandk_outl(n,d,theta_star, n_cont=i)
          np.savetxt(data_path+'run_{}_outl_{}'.format(j,i), X)
#%%
# Load data
datasets = np.zeros((R,outl,n))
for j in range(R):
    for i in range(outl):
        X = np.loadtxt(data_path+'run_{}_outl_{}'.format(j,i))
        datasets[j,i,:] = X
    
# Obtain and save results 
times = []
summary_stats = np.zeros((R,outl, p, 4)) # collect mean, median, mode, st.dev for each bootstrap sample
for j in range(R):
    print("-----Run ", j)
    for n_cont in range(outl):
        print("-----Running for", n_cont*5, "% of outliers-----")
        X =datasets[j,n_cont,:].reshape((n,1))
        npl = NPL.npl(X,B,m,p,l, model = model, model_name = model_name)
        t0 = time.time()
        npl.draw_samples()
        t1 = time.time()
        total = t1- t0
        times.append(total)
        sample = npl.sample
        summary_stats[j,n_cont,:,0] = np.mean(sample, axis=0)
        summary_stats[j,n_cont,:,1] = np.std(sample, axis=0)
        summary_stats[j,n_cont,:,2] = np.median(sample, axis=0)
        summary_stats[j,n_cont,:,3] = stats.mode(sample, axis=0)[0]
        np.savetxt(results_path+'NPL_MMD/thetas_mmd_outl_{}_run_{}.txt'.format(n_cont,j), sample)
        
np.savetxt(results_path+'NPL_MMD/summary_stats.txt', summary_stats)
np.savetxt(results_path+'NPL_MMD/cpu_times.txt', times) 
#%%
#### Reshaping and plotting results for a single run   
# Reshape results
thetas_mmd = np.zeros((p,outl,B))
for i in range(outl):
    for j in range(p):
        sample = np.loadtxt(results_path+'NPL_MMD/thetas_mmd_outl_{}_run_{}.txt'.format(i,r))
        thetas_mmd[j,i,:] = sample[0:500,j]

names = ["A", "B", "g", "k"]
thetas_wabc = np.zeros((p,outl,B))
for n_cont in range(outl):
    for i,name in enumerate(names):
        df = pd.read_csv(results_path+'WABC/thetas_wabc_outl_{}_run_{}new2.csv'.format(n_cont,r))
        thetas_wabc[i,n_cont,:] = df[name]
thetas_wabc[3,:,:] = np.log(thetas_wabc[3,:,:])   # last parameter for gandk is exp(k)
      
# Plot results
fname1 = plots_path+'run_{}_post_marg.png'.format(r)
fname2 = plots_path+'bootstrap_it_densities.png'
fname3 = plots_path+'histograms_sampled_densities.png'
fname4 = plots_path+'sqrt_n_exp.png'

# Figure 4
fig = plt.figure(figsize=(13,5))
gs = gridspec.GridSpec(2, 3)
axes = plot_gnk(B,thetas_wabc, thetas_mmd, theta_star,outl, fname1,save_fig=False)
mg0 = SeabornFig2Grid(axes[0], fig, gs[0], '$\epsilon = 0$', add_title=True)
mg1 = SeabornFig2Grid(axes[1], fig, gs[1], '$\epsilon = 0.05$', add_title=True)
mg2 = SeabornFig2Grid(axes[2], fig, gs[2], '$\epsilon = 0.1$', add_title=True)
mg3 = SeabornFig2Grid(axes[3], fig, gs[3], '0%')
mg4 = SeabornFig2Grid(axes[4], fig, gs[4], '0%')
mg5 = SeabornFig2Grid(axes[5], fig, gs[5], '0%')

gs.tight_layout(fig)
#plt.savefig(fname1)
plt.show()

# Figure 7 in Appendix        
outl = 3
pal = sns.cubehelix_palette(start=.5, rot=-.75, n_colors=B)
fig, ax_array = plt.subplots(1, outl, figsize=(15,5))
for j,ax in enumerate(ax_array):
  for i in range(B):
    theta = thetas_mmd[:,j,i]
    sample = sample_gandk_outl(n,d,theta)
    ax = sns.kdeplot(sample.flatten(), color=pal[i], ax=ax ) 
    ax.set_title(r'$\epsilon = {}$'.format(j*5/100), fontsize=15)
    ax.set_xlim(-5,20)
    ax.set_xlabel('x', fontsize=15)
    if j == 0:
      ax.set_ylabel('Density', fontsize=15)
    else:
      ax.set_ylabel('')
plt.savefig(fname2)

# Figure 6 in Appendix 
theta_posterior_means = np.mean(thetas_mmd, axis=2)
theta_posterior_means_wabc = np.mean(thetas_wabc, axis=2)
f, axes = plt.subplots(2,3,figsize=(15,5))
for i,ax in enumerate(axes.flatten()):
  if i <3:
    X = datasets[0,i,:]
    samples = sample_gandk_outl(n,d,theta_posterior_means[:,i])
    ax.hist(X, bins=100, color='#0072B2', alpha = 0.6, label='Observations') 
    ax.hist(samples, bins=100, color='#D55E00', alpha=0.6, label='Samples')
    ax.set_title(r'$\epsilon = {}$'.format(i*5/100), fontsize=15)
    ax.grid(linewidth=0.5)
    if i == 0:
      ax.legend(fontsize=15)
      ax.set_ylabel('NPL-MMD', fontsize=15)
  else:
    X = datasets[0,(i-3),:]
    samples_wabc = sample_gandk_outl(n,d,theta_posterior_means_wabc[:,(i-3)])
    
    ax.hist(samples_wabc, bins=100, color='#D55E00', alpha=0.6)
    ax.hist(X, bins=100, color='#0072B2', alpha = 0.6) 
    ax.grid(linewidth=0.5)
    if i > 3:
      ax.set_xlim(-60,60)
    else:
      ax.set_ylabel('WABC', fontsize=15)
plt.savefig(fname3)

# Figure 8 in Appendix 
n_range = np.linspace(250,4000,num=10,  dtype=int)
mmd_dists = np.zeros((len(n_range), R))
theta_samples = np.zeros((len(n_range),B,p))
for i, n in enumerate(n_range):
  print(n)
  for r in range(R):
    X = sample_gandk_outl(n,d,theta_star)
    np.savetxt(data_path+'sample_for_divs_20_{}'.format(n), X)
    npl_ = NPL.npl(X,B,m,p,l, model = model, model_name = model_name)
    npl_.draw_samples()
    theta_est = np.mean(npl_.sample, axis=0)
    M = 15000
    x = sample_gandk_outl(M,d,theta_star)
    y = sample_gandk_outl(M,d,theta_est)
    kxx = k(x,x,l)[0]
    kxy = k(y,x,l)[0]
    kyy = k(y,y,l)[0]
    mmd_dists[i,r] = MMD_approx(M,M,kxx,kxy,kyy)
    
    
plt.figure(figsize=(5, 3))

plt.plot(n_range, np.sqrt(np.mean(mmd_dists, axis=1)), label='$\sqrt{\mathbb{E}[\widehat{MMD}^2]}$',marker=11, color='#0072B2')
plt.fill_between(n_range, np.sqrt(np.mean(mmd_dists, axis=1))-np.sqrt(np.std(mmd_dists, axis=1)), np.sqrt(np.mean(mmd_dists, axis=1))+np.sqrt(np.std(mmd_dists, axis=1)), alpha=0.1)

plt.plot(n_range, 2/np.sqrt(n_range), label='$2/\sqrt{n}$', color='#D55E00')
plt.legend()
plt.xlabel('n')
plt.ylabel('$\mathbb{E}[np.sqrt{\widehat{MMD}^2}]$')
plt.savefig(fname4)

#%%
# Sensitivity to hyperparameters experiments (Appendix figures 11-14)

# Sensitivity to alpha
# Range of alphas 
alphas = np.linspace(0.01,300,num=10)

# Sample or Load data 
datasets = np.zeros((outl,n))
for i in range(outl):
    #X = sample_gandk_outl(n,d,theta_star, n_cont=i).reshape((n,))
    #np.savetxt(data_path+'alpha_exp_outl_{}'.format(i), X)
    X = np.loadtxt(data_path+'alpha_exp_outl_{}'.format(i))
    datasets[i,:] = X
    
theta_samples = np.zeros((len(alphas),outl,B,p))
T = 2**11
for i, a in enumerate(alphas):
   print(a)
   for j in range(outl):
     X = datasets[j,:].reshape((n,d))
     npl_ = NPL_prior.npl_prior(X,B,m,s, p,l, model = model, a=a, T=T, model_name = model_name)
     npl_.draw_samples()
     sample = npl_.sample
     theta_samples[i,j,:,:] = sample
     np.savetxt(results_path+'alpha_exp_theta_outl_{}_alpha_{}_T_{}_2.txt'.format(j,a,T), sample)

# MSE plots  
mean_theta_samples = np.mean(theta_samples, axis=2)
def nmse(theta,theta_star):
    mse_ = np.mean(np.asarray((theta-theta_star))**2)/np.mean(theta_star)
    return mse_

mses = np.zeros((len(alphas),outl))
   
for i in range(len(alphas)):
  for j in range(outl):
    mses[i,j] = nmse(mean_theta_samples[i,j,:], theta_star)
            
for j in range(outl):
  plt.plot(alphas, mses[:,j], marker='o', label=r'$\epsilon$ = {}'.format(j*5/100))
  plt.xlabel(r'$\alpha$', fontsize=20)
  plt.ylabel('NMSE', fontsize=20)
  plt.ylim(bottom=0) 
  plt.ylim(top=0.25)
  plt.tick_params(axis='both', which='major', labelsize=15)
  plt.legend(prop={"size":13})
plt.tight_layout() 
plt.savefig(results_path+'alpha_mse.png')

# Densities plots
pal = sns.cubehelix_palette(start=.5, rot=-.75, n_colors=len(alphas))
fig, ax_array = plt.subplots(1, outl, figsize=(20,6))
for j,ax in enumerate(ax_array):
  ax = sns.kdeplot(datasets[0,:].flatten(), ax=ax, linestyle="--", color="black")
  for i in range(len(alphas)):
    theta = mean_theta_samples[i,j,:]
    sample = sample_gandk_outl(n,d,theta, n_cont=0).reshape((n,))
    ax = sns.kdeplot(sample.flatten(), color=pal[i], ax=ax) #label='n={}'.format(n), #, bw_adjust=1
    ax.set_title(r'$\epsilon = {}$'.format(j*5/100), fontsize=20)
    ax.set_xlim(-2,10)
    ax.set_ylim(0,0.4)
    ax.set_xlabel('x', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    if j == 0:
      ax.set_ylabel('Density', fontsize=20)
    else:
      ax.set_ylabel('')

plt.savefig(results_path+'alpha_densities.png')

# Sensitivity to T 
# Range of T 
Ts = [10,100,500,1000,2000,5000]

# Load or sample datasets
datasets = np.zeros((outl,n))
for i in range(outl):
    #X = sample_gandk_outl(n,d,theta_star, n_cont=i).reshape((n,))
    #np.savetxt(data_path+'alpha_exp_outl_{}'.format(i), X)
    X = np.loadtxt(data_path+'alpha_exp_outl_{}'.format(i))
    datasets[i,:] = X

theta_samples = np.zeros((len(Ts),outl,B,p))

a = 0.1  # keep a fixed
for i, T in enumerate(Ts):
   print(T)
   for j in range(outl):
     X = datasets[j,:].reshape((n,d))
     npl_ = NPL_prior.npl_prior(X,B,m,s, p,l, model = model, a=a, T=T, model_name = model_name)
     npl_.draw_samples()
     sample = npl_.sample
     theta_samples[i,j,:,:] = sample
     np.savetxt(results_path+'T_exp_theta_outl_{}_alpha_{}_T_{}_2.txt'.format(j,a,T), sample)

# MSE plot   
mean_theta_samples = np.mean(theta_samples, axis=2)

for i in range(len(Ts)):
  for j in range(outl):
    mses[i,j] = nmse(mean_theta_samples[i,j,:], theta_star)
    
for j in range(outl):
  plt.plot(Ts, mses[:,j], marker='o', label=r'$\epsilon$ = {}'.format(j*5/100))
  plt.xlabel('T', fontsize=20)
  plt.ylabel('NMSE', fontsize=20)
  plt.ylim(bottom=0) 
  plt.ylim(top=0.035)
  plt.tick_params(axis='both', which='major', labelsize=15)
  plt.legend(prop={"size":13})
plt.tight_layout() 
plt.savefig(results_path+'a=0.1_2.png')

# Densities plots
pal = sns.cubehelix_palette(start=.5, rot=-.75, n_colors=len(Ts))
fig, ax_array = plt.subplots(1, outl, figsize=(20,6))
for j,ax in enumerate(ax_array):
  ax = sns.kdeplot(datasets[0,:].flatten(), ax=ax, linestyle="--", color="black")
  for i in range(len(Ts)):
    theta = mean_theta_samples[i,j,:]
    sample = sample_gandk_outl(n,d,theta, n_cont=0).reshape((n,))
    ax = sns.kdeplot(sample.flatten(), color=pal[i], ax=ax) 
    ax.set_title(r'$\epsilon = {}$'.format(j*5/100), fontsize=20)
    ax.set_xlim(-2,10)
    ax.set_ylim(0,0.4)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('x', fontsize=20)
    if j == 0:
      ax.set_ylabel('Density', fontsize=20)
    else:
      ax.set_ylabel('')

plt.savefig(results_path+'T_exps_a=0.1_2.png')

# Sensitivity to l 

# Range of ls
ls = np.array([10**(-1), 10**(-0.5), 10**0, 10**(0.5), 10**1, 10**(1.5), 10**2])

datasets = np.zeros((outl,n))
for i in range(outl):
    #X = sample_gandk_outl(n,d,theta_star, n_cont=i).reshape((n,))
    #np.savetxt(data_path+'alpha_exp_outl_{}'.format(i), X)
    X = np.loadtxt(data_path+'alpha_exp_outl_{}'.format(i))
    datasets[i,:] = X
    
theta_samples = np.zeros((len(ls),outl,B,p))

T = n
a = 0.01
for i, l in enumerate(ls):
   print(l)
   for j in range(outl):
     X = datasets[j,:].reshape((n,d))
     npl_ = npl(X,B,m,s, p,l, model = model, a=a, T=T, model_name = model_name)
     npl_.draw_samples()
     sample = npl_.sample
     theta_samples[i,j,:,:] = sample
     np.savetxt(results_path+'alpha_exp_theta_outl_{}_l_{}_new.txt'.format(j,l), sample)

mses = np.zeros((len(ls),outl))
   
for i in range(len(ls)):
  for j in range(outl):
    mses[i,j] = nmse(mean_theta_samples[i,j,:], theta_star)

# Mse plot
for j in range(outl):
  plt.plot(ls, mses[:,j], label=r'$\epsilon$ = {}'.format(j*5/100), marker='o')
  plt.xscale('log', basex=10)
  plt.xlabel('l', fontsize=20)
  plt.ylabel('NMSE', fontsize=20)
  plt.ylim(bottom=0) 
  plt.ylim(top=40)
  plt.tick_params(axis='both', which='major', labelsize=15)
  plt.legend(prop={"size":13})
plt.tight_layout() 
plt.savefig(results_path+'l_exp_mses_2_new.png')

# MMD loss plot
thetas = np.linspace(-40,40,num=200)
X1 = datasets[0,:].reshape((n,d))
ls = [0.15,1.,10.]
kxx = k(X1,X1,l)[0]
mmds = np.zeros((200,3))
for j,l in enumerate(ls):
  for i,theta in enumerate(thetas):
    thetaa = np.array([3,1,theta,-np.log(2)])
    y = sample_gandk_outl(n,d,thetaa)
    kxy = k(y,X1,l)[0]
    kyy = k(y,y,l)[0]
    mmd = MMD_approx(n,n,kxx,kxy,kyy)
    mmds[i,j] = mmd
    
plt.plot(thetas,mmds[:,0], label='l=0.15')
plt.plot(thetas,mmds[:,1], label='l=0.5')
plt.plot(thetas,mmds[:,2],label='l=1')
plt.plot(thetas,mmds[:,3], label='l=10')
plt.xlabel(r'$\theta_3$')
plt.ylabel(r'$\hat{MMD}$')
plt.axvline(x=1, color='black')
plt.legend()
plt.savefig('losses_theta_3.png')

# Densities plot
pal = sns.cubehelix_palette(start=.5, rot=-.75, n_colors=len(ls))
fig, ax_array = plt.subplots(1, outl, figsize=(20,6))
for j,ax in enumerate(ax_array):
  ax = sns.kdeplot(datasets[0,:].flatten(), ax=ax, linestyle="--", color="black")
  for i in range(len(ls)):
    theta = mean_theta_samples[i,j,:]
    sample = sample_gandk_outl(n,d,theta, n_cont=0).reshape((n,))
    ax = sns.kdeplot(sample.flatten(), color=pal[i], ax=ax) 
    ax.set_title(r'$\epsilon = {}$'.format(j*5/100), fontsize=20)
    ax.set_xlim(-2,12)
    ax.set_ylim(0,0.6)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('x', fontsize=20)
    if j == 0:
      ax.set_ylabel('Density', fontsize=20)
    else:
      ax.set_ylabel('')

plt.savefig(results_path+'l_exps_2_new.png')

