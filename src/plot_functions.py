#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:31:00 2021

"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import math

# Density plots 
def plot_posterior_marginals(B,thetas_wabc, thetas_npl_mmd, thetas_npl_wll, thetas_npl_was, theta_star, n_cont, gaussian=False, save_fig=False):
    """Plot the posterior marginal densities obtained from the parameter sample"""
    if gaussian==True:
        compare = 4
        p = len(theta_star)
        theta_data = np.zeros((p,n_cont,compare*B))
        for i in range(p):
            for j in np.arange(n_cont):
                theta_data[i,j,:] = np.concatenate((thetas_wabc[i,j,:],thetas_npl_mmd[i,j,:],thetas_npl_wll[i,j,:],thetas_npl_was[i,j,:]))
            
        model_name_data = np.concatenate((['WABC']*B,['NPL-MMD']*B,['NPL-WLL']*B,['NPL-WAS']*B))
    else:
        compare = 2
        p = len(theta_star)
        theta_data = np.zeros((p,n_cont,compare*B))
        for i in range(p):
            for j in np.arange(n_cont):
                theta_data[i,j,:] = np.concatenate((thetas_wabc[i,j,:],thetas_npl_mmd[i,j,:]))
            
        model_name_data = np.concatenate((['WABC']*B,['NPL-MMD']*B))

    columns = []
    for i in range(n_cont*p):
         columns.append('theta_{}'.format(i))  
    columns.append('model')
    theta_data = np.reshape(theta_data,(n_cont*p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(n_cont*p) }) 
    df_all['model'] =  model_name_data
    fig, ax_array = plt.subplots(p, n_cont, figsize=(10,30))
    titles = ['{} % of outliers'.format(i*5) for i in range(n_cont)]
    
    for ax, i in zip(ax_array.flatten(), range(0, p * n_cont)):
        ax = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="model", multiple="layer", shade=True, common_norm=False)
        ax.set_xlabel('theta {}'.format(math.floor(i/n_cont)+1))
        #ax.set_title(titles[i]) #i%3
        ax.axvline(theta_star[i], color='black')  #prepei na ftiaxw ta indexes  #math.floor(i/n_cont)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if save_fig == True:
        fig.savefig('posterior_marginal_plot_togswitch.png')
      
    return fig, ax_array

def plot_mse(thetas_wabc, thetas_npl_mmd, thetas_npl_wll, thetas_npl_was, mse_MLE, theta_star, n_cont, gaussian=False, save_fig=False):
    """Plots the mse error for each parameter against 
    the percentage of outliers in the data"""
    
    # Function to calculate mse 
    def mse(thetas,theta_star):
        mse_ = np.mean(np.asarray((thetas-theta_star))**2)
        return mse_
    
    p = len(theta_star)
    mse_ABC = np.zeros((n_cont,p))
    mse_MMD = np.zeros((n_cont,p))
    mse_WLL = np.zeros((n_cont,p))
    mse_WAS = np.zeros((n_cont,p))
   
    for i in range(n_cont):
        for j in range(p):
            mse_ABC[i,j] = mse(thetas_wabc[j,i,:], theta_star[j])
            mse_MMD[i,j] = mse(thetas_npl_mmd[j,i,:], theta_star[j])
            mse_WLL[i,j] = mse(thetas_npl_wll[j,i,:], theta_star[j])
            mse_WAS[i,j] = mse(thetas_npl_was[j,i,:], theta_star[j])
            
    #print(mse_ABC)
    #print(mse_MMD)
    #print(mse_WLL)
    fig, ax_array = plt.subplots(p, 1, figsize=(5,10))
    if p>1:
        for j,ax in enumerate(ax_array):
            ax.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_MMD[:,j], 'r-', label='NPL-MMD')
            ax.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_ABC[:,j], 'b-', label='W-ABC')
            if gaussian == True:
                ax.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_WLL[:,j], 'g-', label='NPL-WLL')
                ax.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_WAS[:,j], 'y-', label='NPL-WAS')
                ax.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_MLE[:,j], 'c-', label='MLE')
            ax.legend(loc='best')
            #ax.get_frame().set_alpha(0.5)
            ax.set_xlabel("Percentage of outliers",fontsize='x-large')
            ax.set_ylabel("MSE",fontsize='x-large')
            ax.set_title('theta {}'.format(j+1))
    else:
        ax_array.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_MMD[:,j], 'r-', label='NPL-MMD')
        ax_array.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_ABC[:,j], 'b-', label='W-ABC')
        if gaussian == True:
            ax_array.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_WLL[:,j], 'g-', label='NPL-WLL')
            ax_array.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_WAS[:,j], 'y-', label='NPL-WAS')
            ax_array.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_MLE[:,j], 'c-', label='MLE')
        ax_array.legend(loc='best', ncol=1)
        #ax.get_frame().set_alpha(0.5)
        ax_array.set_xlabel('Percentage of outliers',fontsize='x-large')
        ax_array.set_ylabel('MSE',fontsize='x-large')
        ax_array.set_title('theta {}'.format(j+1))
    if save_fig == True:
        fig.savefig('mse_gandk.png')
    fig.tight_layout() 
    return fig, ax_array


#fig1.savefig('mse_prelim.png')
    
    