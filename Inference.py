#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:40:28 2021

@author: HaritaDellaporta
"""

from utils import sample_gaussian_outl, sample_gandk_outl, k, gen_gandk
import NPL
import models
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
# testing
#%% 
# Import data from R
#data = pyreadr.read_r('mvnormaldata.d1.n200.RData')
#theta_star = np.array(data["true_theta"])
##theta_star = np.squeeze(np.asarray(data["true_theta"])) # shape (d,)
#X = np.asarray(data['obs']).transpose() # shape (n,d)

# Export data to use in R - Gaussian model
np.random.seed(11)
n = 200
d = 2
theta_star = np.ones(d)
s = 1
outl = 3
#for i in range(outl):
#    X = sample_gaussian_outl(n,d,s,theta_star, n_cont=i)
#    np.savetext("data_{}_{}.txt".format(i,n))
X1 = sample_gaussian_outl(n,d,s,theta_star,n_cont=0) 
X2 = sample_gaussian_outl(n,d,s,theta_star,n_cont=1) 
X3 = sample_gaussian_outl(n,d,s,theta_star,n_cont=2) 
np.savetxt("data_0_{}.txt".format(n), X1)
np.savetxt("data_1_{}.txt".format(n), X2)
np.savetxt("data_2_{}.txt".format(n), X3)
#%%
# Export data to use in R - g-and-k model
#np.random.seed(11)
#outl = 3
#theta_star = np.array([3,1,1,-np.log(2)]) 
#n = 2**11
#d = 1
#for i in range(outl):
#    X = sample_gandk_outl(n,d,theta_star, n_cont=i)
#    np.savetxt("data_{}_{}_t.txt".format(i,n),X)
#X1 = sample_gandk_outl(n,d,theta_star,n_cont=0) 
#X2 = sample_gandk_outl(n,d,theta_star,n_cont=1) 
#X3 = sample_gandk_outl(n,d,theta_star,n_cont=2) 
#np.savetxt("data_0_gandk_{}.txt".format(n), X1)
#np.savetxt("data_1_gandk_500.txt", X2)
#np.savetxt("data_2_gandk_500.txt", X3)
#%%
# Set parameters
np.random.seed(11)
n = 200
m = 2**9
d = 2
#theta_star = np.array([3,1,1,-np.log(2)]) 
#X = sample_gandk_outl(n,d,theta_star, n_cont=0)
#np.savetxt("data_{}_{}_megalon.txt".format(0,n), X)
theta_star = np.ones(d)
s = 1  
l = -1
# dimensions of data
p = 2
B = 512
outl = 1
method = 'SGD'
model_name = 'gaussian'
model = models.gauss_model(m,d,s)
#X1 = np.reshape(np.loadtxt('data_0_2048_new.txt'), (2048,1))
#%% optimisation functions

def optim_gaus(X1,X2,X3,n,m,s,l,theta_star,d,p,B,outl,method,model_name,model): 
    results = np.zeros((3,p,outl)) # each of the rows are MMD, MLE, WLL resp.
    samples = np.zeros((outl,B,p))
#thetas = np.zeros((B,p))
    for n_cont in np.arange(outl):
        print("-----Running for", n_cont*5, "% of outliers-----")
        #X = sample_gaussian_outl(n,d,s,theta_star,n_cont=n_cont) # prepei na xrisimopoiw ta idia data kai gia tin alli methodo!!!
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
        #mmd_loss = npl.mmd_loss
        #np.savetxt("current_sample.txt", sample)
        np.savetxt('outl_{}_t.txt'.format(n_cont), sample)
        #sample_mmd = np.mean(sample, axis=0)
        sample_mle = np.mean(X, axis=0)
        #sample_wll = np.mean(wll_sample, axis=0)
      
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
        np.savetxt('outl_{}_ll.txt'.format(n_cont+3), sample)
    #thetas = npl.thetas
    #mmd_loss = npl.mmd_loss
        
    #sample_mmd = np.mean(sample, axis=0)
        for i in range(p):
            results[i,n_cont] = mean_squared_error(theta_star[i]*np.ones(B), sample[:,i])
        
    return results, samples
    

#%%
#X1 = sample_gandk_outl(n,d,theta_star,n_cont=0) 
X1 = np.reshape(np.loadtxt('data_0_200.txt'), (n,d))
X2 = np.reshape(np.loadtxt('data_1_200.txt'), (n,d))
X3 = np.reshape(np.loadtxt('data_2_200.txt'), (n,d))
    #%%
# Optimise 
results_gauss, samples_gauss = optim_gaus(X1,X2,X3,n,m,s,l,theta_star,d,p,B,3,method,model_name,model)
#%%
#np.savetxt(fname='res_500_gandk.txt', X=results_gandk.reshape((-1,3)))
#np.savetxt(fname='sam_500_gandk.txt', X=samples_gandk[0,:,:])
#%%
################################################ Plots ########################################################
fig1 = plt.figure(figsize=(6, 6))

plt.plot(np.linspace(0,4*outl,outl), results[0,:], 'r--', label='MMD')
plt.plot(np.linspace(0,4*outl,outl), results[1,:], 'b--', label='MLE')
plt.plot(np.linspace(0,4*outl,outl), results[2,:], 'g--', label='WLL')
leg = plt.legend(loc='best', ncol=1)
leg.get_frame().set_alpha(0.5)
plt.xlabel('Percentage of outliers',fontsize='x-large')
plt.ylabel('MSE',fontsize='x-large')

fig1.set_size_inches(12, 4)

plt.tight_layout()

#fig1.savefig('mse.png')
#%%
fig2 = plt.figure(figsize=(6, 6))
plt.plot(range(B), abs(mmd_loss), label='MMD')
leg = plt.legend(loc='best', ncol=1)
leg.get_frame().set_alpha(0.5)
plt.xlabel('Bootstrap iteration',fontsize='x-large')
plt.ylabel('MMD^2 loss',fontsize='x-large')
fig2.show()
#fig2.savefig('loss.png')
#%%
fig3 = plt.figure(figsize=(8, 4))
plt.subplot(1,2,1)
plt.scatter(sample[:,0], sample[:,1])
plt.scatter(theta_star[0],theta_star[1], marker='+')
#plt.plot(np.linspace(0,4*outl,outl), results[1,:], 'b--', label='MLE')
#plt.plot(np.linspace(0,4*outl,outl), results[2,:], 'g--', label='WLL')
#leg = plt.legend(loc='best', ncol=1)
#leg.get_frame().set_alpha(0.5)
plt.xlabel('theta_1',fontsize='x-large')
plt.ylabel('theta_2',fontsize='x-large')

plt.subplot(1,2,2)
plt.scatter(sample[:,2], sample[:,3])
plt.scatter(theta_star[2],theta_star[3], marker='+')
#plt.plot(np.linspace(0,4*outl,outl), results[1,:], 'b--', label='MLE')
#plt.plot(np.linspace(0,4*outl,outl), results[2,:], 'g--', label='WLL')
#leg = plt.legend(loc='best', ncol=1)
#leg.get_frame().set_alpha(0.5)
plt.xlabel('theta_3',fontsize='x-large')
plt.ylabel('theta_4',fontsize='x-large')
fig3.set_size_inches(12, 4)

plt.tight_layout()
#%%
### Create arrays with sampled parameters to pass to plot functions
outl=3
B = 512
# Get W-ABC results from R
df1 = pd.read_csv('results_0_l_2048.csv')
df2 = pd.read_csv('results_1_l_2048.csv')
df3 = pd.read_csv('results_2_l_2048.csv')
names = ["A", "B", "g", "k"]
thetas_wabc_gandk = np.zeros((p,outl,B))
for n_cont in range(outl):
    for i,name in enumerate(names):
        if n_cont == 0:
            df = df1
        elif n_cont == 1:
            df = df2
        elif n_cont == 2:
            df = df3
        thetas_wabc_gandk[i,n_cont,:] = df[name]
thetas_wabc_gandk[3,:,:] = np.log(thetas_wabc_gandk[3,:,:])
        
#%%
sample_1 = np.loadtxt('outl_2_gandk_g_2048_bfgs_l.txt')
sample_2 = np.loadtxt('outl_3_gandk_g_2048_bfgs_l.txt')
sample_3 = np.loadtxt('outl_4_gandk_g_2048_bfgs_l.txt')
thetas_mmd_gandk = np.zeros((p,outl,B))
for n_cont in range(outl):
    for j in range(p):
        if n_cont == 0:
            sample = sample_1
        elif n_cont == 1:
            sample = sample_2
        elif n_cont == 2:
            sample = sample_3
        thetas_mmd_gandk[j,n_cont,:] = sample[:,j]
#%%
for i in range(outl):
    for j in range(p):
        ind_array = np.where(abs(thetas_mmd_gandk[j,i,:] - np.mean(thetas_mmd_gandk[j,i,:])) > np.std(thetas_mmd_gandk[j,i,:]))
        for idx in ind_array:
            thetas_mmd_gandk[j,i,idx] = theta_star[j] # ftiaxto
#%%
# Density plots 
theta_0_data = np.concatenate((theta_abc_0,samples[0,:,0]))
theta_1_data = np.concatenate((theta_abc_1,samples[1,:,0]))
theta_2_data = np.concatenate((theta_abc_2,samples[2,:,0]))
#theta_2_data = np.concatenate((theta2_abc,samples[0,:,1]))
model_name_data = np.concatenate((['WABC']*512,['NPL-MMD']*512))
#np_data = theta_1_data, theta_2_data, model_name_data
columns = ['theta_0', 'theta_1', 'theta_2', 'model'] # ,'theta2'
df_all = pd.DataFrame(
    {'theta_0': theta_0_data,
     'theta_1': theta_1_data,
     'theta_2': theta_2_data,
     #'theta2': theta_2_data,
     'model': model_name_data}, columns=columns)
#d = {'theta_1': [np.concatenate((theta1_abc,sample[:,0]))], 'theta_2':[np.concatenate((theta2_abc,sample[:,1]))], 'Model': [np.concatenate((['WABC'*512],['NPL-MMD'*500]))]}
#df_all = pd.DataFrame(data=np_data, columns=["theta_1", "theta_2", "method"])
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]})
titles = ['theta_0_outliers', 'theta_1_outliers', 'theta_2_outliers']

ax1 = sns.kdeplot(data=df_all, x="theta_0", linestyle="-", linewidth = 4, ax=ax1, alpha=0.3,hue="model", multiple="stack")
ax1.set_title(titles[0])
ax1.axvline(theta_star[0], color='black')
ax2 = sns.kdeplot(data=df_all, x="theta_1", linestyle="-", linewidth = 4, ax=ax2, alpha=0.3,hue="model", multiple="stack")
ax2.set_title(titles[1])
ax2.axvline(theta_star[0], color='black')
ax3 = sns.kdeplot(data=df_all, x="theta_2", linestyle="-", linewidth = 4, ax=ax3, alpha=0.3,hue="model", multiple="stack")
ax3.set_title(titles[2])
ax3.axvline(theta_star[0], color='black')


#plt.savefig("fig_200_outliers.jpg")
#%%
# Density plots 
theta_1_data = np.concatenate((theta1_abc,samples[0,:,0]))
#theta_2_data = np.concatenate((theta2_abc,samples[0,:,1]))
model_name_data = np.concatenate((['WABC']*512,['NPL-MMD']*B))
#np_data = theta_1_data, theta_2_data, model_name_data
columns = ['theta1','model'] # ,'theta2'
df_all = pd.DataFrame(
    {'theta1': theta_1_data,
     #'theta2': theta_2_data,
     'model': model_name_data}, columns=columns)
#d = {'theta_1': [np.concatenate((theta1_abc,sample[:,0]))], 'theta_2':[np.concatenate((theta2_abc,sample[:,1]))], 'Model': [np.concatenate((['WABC'*512],['NPL-MMD'*500]))]}
#df_all = pd.DataFrame(data=np_data, columns=["theta_1", "theta_2", "method"])
fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
titles = ['theta_1']#, 'theta_2']

ax1 = sns.kdeplot(data=df_all, x="theta1", linestyle="-", linewidth = 4, ax=ax1, alpha=0.3,hue="model", multiple="stack")
ax1.set_title(titles[0])
ax1.axvline(theta_star[0], color='black')
#ax2 = sns.kdeplot(data=df_all, x="theta2", linestyle="-", linewidth = 4, ax=ax2, alpha=0.3,hue="model", multiple="stack")
#ax2.set_title(titles[1])
#ax2.axvline(theta_star[1], color='black')

#plt.savefig("fig_200.jpg")
#%%
from scipy.stats import norm

# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 10, 0.001)
# Mean = 0, SD = 2.
plt.plot(x_axis, norm.pdf(x_axis,theta_star[0],1))
plt.show()
#%%
#plt.hist2d(X[0,:],X[0,:] )
import scipy.stats as stats

mu = theta_star[0][0]
sigma = 1
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
#x = range(int(X.max()+1))
plt.hist(X,bins=25, density=True, alpha=0.6, color='g')
plt.plot(x, stats.norm.pdf(x, mu, sigma), color = 'b', label = 'P_0')
plt.plot(x, stats.norm.pdf(x, np.mean(samples) , sigma), color = 'r', label = 'NPL-MMD')
plt.plot(x, stats.norm.pdf(x, np.mean(theta1_abc), sigma), color = 'y',label = 'W-ABC')
plt.legend()
