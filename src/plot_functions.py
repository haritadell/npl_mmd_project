#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:31:00 2021

@author: HaritaDellaporta
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import math

# Density plots 
def plot_posterior_marginals(B,thetas_wabc, thetas_npl_mmd, theta_star, n_cont, save_fig=False):
    """Plot the posterior marginal densities obtained from the parameter sample"""
    p = len(theta_star)
    theta_data = np.zeros((p,n_cont,2*B))
    for i in range(p):
        for j in np.arange(n_cont):
            theta_data[i,j,:] = np.concatenate((thetas_wabc[i,j,:],thetas_npl_mmd[i,j,:]))
            
    model_name_data = np.concatenate((['WABC']*B,['NPL-MMD']*B))

    columns = []
    for i in range(n_cont*p):
         columns.append('theta_{}'.format(i))  
    columns.append('model')
    theta_data = np.reshape(theta_data,(n_cont*p,2*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(n_cont*p) }) 
    df_all['model'] =  model_name_data
    fig, ax_array = plt.subplots(p, n_cont, figsize=(10,10))
    titles = ['{} % of outliers'.format(i*5) for i in range(n_cont)]
    
    for ax, i in zip(ax_array.flatten(), range(0, p * n_cont)):
        ax = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="model", multiple="stack")
        ax.set_xlabel('theta {}'.format(math.floor(i/n_cont)+1))
        ax.set_title(titles[i%3])
        ax.axvline(theta_star[math.floor(i/n_cont)], color='black')  #prepei na ftiaxw ta indexes 
        
    if save_fig == True:
        fig.savefig('posterior_marginal_plot_gauss_200.png')
    fig.tight_layout()  
    return fig, ax_array

def plot_mse(thetas_wabc, thetas_npl_mmd, theta_star, n_cont, save_fig=False):
    """Plots the mse error for each parameter against 
    the percentage of outliers in the data"""
    
    # Function to calculate mse 
    def mse(thetas,theta_star):
        mse_ = np.mean(np.asarray((thetas-theta_star))**2)
        return mse_
    
    p = len(theta_star)
    mse_ABC = np.zeros((n_cont,p))
    mse_MMD = np.zeros((n_cont,p))
    for i in range(n_cont):
        for j in range(p):
            mse_ABC[i,j] = mse(thetas_wabc[j,i,:], theta_star[j])
            mse_MMD[i,j] = mse(thetas_npl_mmd[j,i,:], theta_star[j])
    print(mse_ABC)
    print(mse_MMD)
    fig, ax_array = plt.subplots(p, 1, figsize=(10,10))
    for j,ax in enumerate(ax_array):
        ax.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_MMD[:,j], 'r-', label='NPL-MMD')
        ax.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_ABC[:,j], 'b--', label='W-ABC')
        ax.legend(loc='best', ncol=1)
        #ax.get_frame().set_alpha(0.5)
        ax.set_xlabel('Percentage of outliers',fontsize='x-large')
        ax.set_ylabel('MSE',fontsize='x-large')
        ax.set_title('theta {}'.format(j+1))
    if save_fig == True:
        fig.savefig('mse_plot_gauss_200.png')
    fig.tight_layout() 
    return fig, ax_array


#fig1.savefig('mse_prelim.png')
    
    