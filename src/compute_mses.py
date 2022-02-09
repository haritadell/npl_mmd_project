#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:56:02 2021

@author: HaritaDellaporta
"""


import numpy as np
import pandas as pd
from utils import mse

### This script loads results and computes normalized MSE over multiple runs ###
# Gaussian 

data_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/Gaussian_location_model/"
results_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/Gaussian_location_model/"
plots_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/Plots/"

R = 10 
outl = 3
p = 4
d = 4
theta_star = np.ones(d)
B = 500

estimators_wabc = np.zeros((R,outl,p))
for r in range(R):
    names = ["X1", "X2", "X3", "X4"]
    thetas_wabc = np.zeros((p,outl,B))
    for n_cont in range(outl):
        for i,name in enumerate(names):
            df = pd.read_csv(results_path+'WABC/thetas_wabc_outl_{}_run_{}_dim_{}.csv'.format(n_cont,r,d))
            thetas_wabc[i,n_cont,:] = df[name]
    estimators_wabc[r,:,:] = np.mean(thetas_wabc, axis=2).transpose()
    
estimators_mabc = np.zeros((R,outl,p))
for r in range(R):
    names = ["X1", "X2", "X3", "X4"]
    thetas_mabc = np.zeros((p,outl,B))
    for n_cont in range(outl):
        for i,name in enumerate(names):
            df = pd.read_csv(results_path+'WABC/thetas_mabc_outl_{}_run_{}_dim_{}.csv'.format(n_cont,r,d))
            df = df[0:500]
            thetas_mabc[i,n_cont,:] = df[name]
    estimators_mabc[r,:,:] = np.mean(thetas_mabc, axis=2).transpose()
    
estimators_was = np.zeros((R,outl,p))
for r in range(R):
  thetas_was = np.zeros((p,outl,B))
  for n_cont in range(outl):
    for j in range(p):
        sample = np.loadtxt(results_path+'NPL_WAS/thetas_mmd_outl_{}_run_{}_dim_{}_.txt'.format(n_cont,r,d))
        thetas_was[j,n_cont,:] = sample[:,j]
    estimators_was[r,:,:] = np.mean(thetas_was, axis=2).transpose()

summary_stats_mmd = np.loadtxt(results_path+'NPL_MMD/summary_stats.txt').reshape((R,outl, p, 4))
summary_stats_wll = np.loadtxt(results_path+'NPL_WLL/summary_stats.txt').reshape((R,outl, p, 4))

estimators_mmd = summary_stats_mmd[:,:,:,0]
mean_square_errors_mmd = np.zeros((R,outl))
estimators_wll = summary_stats_wll[:,:,:,0]
mean_square_errors_wll = np.zeros((R,outl))
mean_square_errors_wabc = np.zeros((R,outl))
mean_square_errors_was = np.zeros((R,outl))
mean_square_errors_mabc = np.zeros((R,outl))


for r in range(R):
    for i in range(outl):
        mean_square_errors_mmd[r,i] = mse(estimators_mmd[r,i,:], theta_star)
        mean_square_errors_wll[r,i] = mse(estimators_wll[r,i,:], theta_star)
        mean_square_errors_wabc[r,i] = mse(estimators_wabc[r,i,:], theta_star)
        mean_square_errors_was[r,i] = mse(estimators_was[r,i,:], theta_star)
        mean_square_errors_mabc[r,i] = mse(estimators_mabc[r,i,:], theta_star)
    
averages = np.mean(mean_square_errors_mmd, axis=0)
stds = np.std(mean_square_errors_mmd, axis=0)
averages_wll = np.mean(mean_square_errors_wll, axis=0)
stds_wll = np.std(mean_square_errors_wll, axis=0)
averages_wabc = np.mean(mean_square_errors_wabc, axis=0)
stds_wabc = np.std(mean_square_errors_wabc, axis=0)
averages_was = np.mean(mean_square_errors_was, axis=0)
stds_was = np.std(mean_square_errors_was, axis=0)
averages_mabc = np.mean(mean_square_errors_mabc, axis=0)
stds_mabc = np.std(mean_square_errors_mabc, axis=0)

print("Mean square erros for mmd", averages)
print("Std of square erros for mmd", stds)
print("Mean square erros for wabc", averages_wabc)
print("Std of square erros for wabc", stds_wabc)
print("Mean square erros for wll", averages_wll)
print("Std of square erros for wll", stds_wll)
print("Mean square erros for was", averages_was)
print("Std of square erros for was", stds_was)
print("Mean square erros for mabc", averages_mabc)
print("Std of square erros for mabc", stds_mabc)

times = np.zeros(R*outl)
count = 0
for r in range(R):
    for i in range(outl):
        name = ["time"]
        df = pd.read_csv(results_path+'WABC/thetas_wabc_outl_{}_run_{}_dim_{}.csv'.format(i,r,4))
        times[count] = df[name].iloc[0]
        count += 1
print('WABC time', np.mean(times))

cpu_times = np.mean(np.loadtxt(results_path+'NPL_MMD/cpu_times.txt'))
print('MMD cpu time',cpu_times)

gpu_times = np.mean(np.loadtxt(results_path+'NPL_MMD/times_gauss.txt'))
print('MMD gpu time',gpu_times)

#%% 
# G-and-k
# Set paths
data_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/G_and_k_model/"
results_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/G_and_k_model/"
plots_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/Plots/G_and_k/"

R = 10 
outl = 3
p = 4
d = 1
theta_star = np.array([3,1,1,-np.log(2)]) 
B = 500

estimators_wabc = np.zeros((R,outl,p))
for r in range(R):
    names = ["A", "B", "g", "k"]
    thetas_wabc = np.zeros((p,outl,B))
    for n_cont in range(outl):
        for i,name in enumerate(names):
            df = pd.read_csv(results_path+'WABC/thetas_wabc_outl_{}_run_{}new2.csv'.format(n_cont,r))
            thetas_wabc[i,n_cont,:] = df[name]
    thetas_wabc[3,:,:] = np.log(thetas_wabc[3,:,:])   # last parameter for gandk is exp(k)
    estimators_wabc[r,:,:] = np.mean(thetas_wabc, axis=2).transpose()

summary_stats_mmd = np.loadtxt(results_path+'NPL_MMD/summary_stats.txt').reshape((R,outl, p, 4))

estimators_mmd = summary_stats_mmd[:,:,:,0]
mean_square_errors_mmd = np.zeros((R,outl))
mean_square_errors_wabc = np.zeros((R,outl))

for r in range(R):
    for i in range(outl):
        mean_square_errors_mmd[r,i] = mse(estimators_mmd[r,i,:], theta_star)
        mean_square_errors_wabc[r,i] = mse(estimators_wabc[r,i,:], theta_star)

averages = np.mean(mean_square_errors_mmd, axis=0)
stds = np.std(mean_square_errors_mmd, axis=0)
averages_wabc = np.mean(mean_square_errors_wabc, axis=0)
stds_wabc = np.std(mean_square_errors_wabc, axis=0)

print("Mean square erros for mmd", averages)
print("Std of square erros for mmd", stds)
print("Mean square erros for wabc", averages_wabc)
print("Std of square erros for wabc", stds_wabc)

times = np.zeros(R*outl)
count = 0
for r in range(R):
    for i in range(outl):
        name = ["time"]
        df = pd.read_csv(results_path+'WABC/thetas_wabc_outl_{}_run_{}new2.csv'.format(i,r))
        times[count] = df[name].iloc[0]
        count += 1
print('WABC time', np.mean(times))

cpu_times = np.mean(np.loadtxt(results_path+'NPL_MMD/cpu_times.txt'))
print('MMD cpu time',cpu_times)

gpu_times = np.mean(np.loadtxt(results_path+'NPL_MMD/times_gnk_.txt'))
print('MMD gpu time',gpu_times)
#%% 
# Toggle Switch model
# Set paths
data_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/Toggle_switch_model/"
results_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/Toggle_switch_model/"
plots_path = "/Users/HaritaDellaporta/Dropbox/mmd_project_code/Plots/Toggle_switch/" 

R = 5
p = 7
d = 1
theta_star = np.array([22.,12.,4.,4.5,325.,0.25,0.15]) 
B = 300

estimators_wabc = np.zeros((R,p))
for r in range(R):
    names = ["alpha_1", "alpha_2", "beta_1", "beta_2", "mu", "sigma", "gamma"]
    thetas_wabc = np.zeros((p,B))
    for i,name in enumerate(names):
        df = pd.read_csv(results_path+'WABC/thetas_wabc_run_{}.csv'.format(r))
        thetas_wabc[i,:] = df[name]
    estimators_wabc[r,:] = np.mean(thetas_wabc, axis=1)

estimators_mmd = np.zeros((R,p))
for r in range(R):
    sample = np.loadtxt(results_path+'NPL_MMD/thetas_mmd_run_{}.txt'.format(r))
    sample[:,0] = np.exp(sample[:,0])
    sample[:,1] = np.exp(sample[:,1])
    sample[:,4] = np.exp(sample[:,4])
    sample[:,5] = np.exp(sample[:,5])
    estimators_mmd[r,:] = np.mean(sample, axis=0)

mean_square_errors_mmd = np.zeros(R)
mean_square_errors_wabc = np.zeros(R)

for r in range(R):
    mean_square_errors_mmd[r] = mse(estimators_mmd[r,:], theta_star)
    mean_square_errors_wabc[r] = mse(estimators_wabc[r,:], theta_star)
    
averages = np.mean(mean_square_errors_mmd, axis=0)
stds = np.std(mean_square_errors_mmd, axis=0)
averages_wabc = np.mean(mean_square_errors_wabc, axis=0)
stds_wabc = np.std(mean_square_errors_wabc, axis=0)

print("Mean square erros for mmd", averages)
print("Std of square erros for mmd", stds)
print("Mean square erros for wabc", averages_wabc)
print("Std of square erros for wabc", stds_wabc)

times = np.zeros(R)
for r in range(R):
    name = ["time"]
    df = pd.read_csv(results_path+'WABC/thetas_wabc_run_{}.csv'.format(r))
    times[r] = df[name].iloc[0]

print('WABC time', np.mean(times))

gpu_times = np.mean(np.loadtxt(results_path+'NPL_MMD/times_togswitch.txt'))
print('MMD gpu time',gpu_times)










