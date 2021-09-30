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
def plot_posterior_marginals(B,thetas_wabc, thetas_npl_mmd, thetas_npl_wll, thetas_npl_was, theta_star, n_cont, fname, gaussian=False, save_fig=False):
    """Plot the posterior marginal densities obtained from the parameter sample"""
    if gaussian==True:
        compare = 3
        p = len(theta_star)
        theta_data = np.zeros((p,n_cont,compare*B))
        for i in range(p):
            for j in np.arange(n_cont):
                theta_data[i,j,:] = np.concatenate((thetas_wabc[i,j,:],thetas_npl_mmd[i,j,:],thetas_npl_wll[i,j,:])) #,thetas_npl_was[i,j,:]))
            
        model_name_data = np.concatenate((['WABC']*B,['NPL-MMD']*B,['NPL-WLL']*B)) #,['NPL-WAS']*B))
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
    fig, ax_array = plt.subplots(p, n_cont, figsize=(10,30))  #(20,8)
    titles = ['{} % of outliers'.format((i)*5) for i in range(n_cont)]
    colors = ['#0072B2', '#D55E00', '#009E73'] 
    
    for ax, i in zip(ax_array.flatten(), range(0, p * n_cont)):
        ax = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="model", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False)
        ax.set_xlabel('theta {}'.format(math.floor(i/n_cont)+1))
        #ax.set_title(titles[i%6]) #i%3
        ax.axvline(theta_star[math.floor(i/n_cont)], color='black')  #prepei na ftiaxw ta indexes  #
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if save_fig == True:
        fig.savefig(fname)
      
    return fig, ax_array

def plot_mse(thetas_wabc, thetas_npl_mmd, thetas_npl_wll, thetas_npl_was, mse_MLE, theta_star, n_cont, fname, gaussian=False, save_fig=False):
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
            
    fig, ax_array = plt.subplots(1, p, figsize=(20,5))
    if p>1:
        for ax,j in zip(ax_array.flatten(), range(0, 4)):
            ax.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_MMD[:,j], linestyle='--', marker='o', color='b', label='NPL-MMD')
            ax.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_ABC[:,j], linestyle='--', marker='o', color='r', label='W-ABC')
            if gaussian == True:
                ax.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_WLL[:,j], 'g-', label='NPL-WLL')
                ax.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_WAS[:,j], 'y-', label='NPL-WAS')
                ax.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_MLE[:,j], 'c-', label='MLE')
            ax.legend(loc='best')
            ax.set_xlabel("Percentage of outliers",fontsize='x-large')
            ax.set_ylabel("MSE",fontsize='x-large')
            ax.set_title('theta {}'.format(j+1))
    else:
        ax_array.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_MMD[:,j], linestyle='--', marker='o', color='b', label='NPL-MMD')
        ax_array.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_ABC[:,j], linestyle='--', marker='o', color='r', label='W-ABC')
        if gaussian == True:
            ax_array.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_WLL[:,j], 'g-', label='NPL-WLL')
            #ax_array.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_WAS[:,j], 'y-', label='NPL-WAS')
            #ax_array.plot(np.linspace(0,5*(n_cont-1),n_cont), mse_MLE[:,j], 'c-', label='MLE')
        ax_array.legend(loc='best', ncol=1)
        #ax.get_frame().set_alpha(0.5)
        ax_array.set_xlabel('Percentage of outliers',fontsize='x-large')
        ax_array.set_ylabel('MSE',fontsize='x-large')
        #ax_array.set_title('theta {}'.format(j+1))
        ax_array.set_title('MSE for the mean of the univariate Gaussian distribution')
    if save_fig == True:
        fig.savefig(fname)
    fig.tight_layout() 
    return fig, ax_array


#fig1.savefig('mse_prelim.png')
def plot_bivariate_gaussian(B,thetas_wabc, thetas_mmd, thetas_wll, theta_star, n_cont, fname, save_fig=False):
    compare = 3
    p = len(theta_star)
    theta_data = np.zeros((p,n_cont,compare*B))
    for i in range(p):
        for j in np.arange(n_cont):
            theta_data[i,j,:] = np.concatenate((thetas_wabc[i,j,:],thetas_mmd[i,j,:],thetas_wll[i,j,:]))
            
    model_name_data = np.concatenate((['WABC']*B,['NPL-MMD']*B,['NPL-WLL']*B))
    columns = []
    for i in range(n_cont*p):
         columns.append('theta_{}'.format(i))  
    columns.append('model')
    theta_data = np.reshape(theta_data,(n_cont*p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(n_cont*p) }) 
    df_all['model'] =  model_name_data
    titles = ['{} % of outliers'.format(i*5) for i in range(n_cont)]
    
    ax1 = sns.jointplot(data=df_all, x="theta_0", y="theta_3", hue="model", kind='kde', shade=True, multiple="layer",alpha=0.6)
    ax1.set_axis_labels('theta_1', 'theta_2')
    ax1.ax_joint.axvline(x=theta_star[0], color='black', linestyle='--', linewidth = 1)  
    ax1.ax_joint.axhline(y=theta_star[0], color='black', linestyle='--', linewidth = 1)  
    ax1.fig.suptitle(titles[0])
    ax1.fig.tight_layout()
    ax1.fig.subplots_adjust(top=0.95)
    if save_fig:
        plt.savefig(fname+'_0.png')
    
    ax2 = sns.jointplot(data=df_all, x="theta_1", y="theta_4", hue="model", kind='kde', shade=True, multiple="layer",alpha=0.6)
    ax2.set_axis_labels('theta_1', 'theta_2')
    ax2.ax_joint.axvline(x=theta_star[0], color='black', linestyle='--', linewidth = 1)  
    ax2.ax_joint.axhline(y=theta_star[0], color='black', linestyle='--', linewidth = 1)  
    ax2.fig.suptitle(titles[1])
    ax2.fig.tight_layout()
    ax2.fig.subplots_adjust(top=0.95)
    if save_fig:
        plt.savefig(fname+'_1.png')
    
    ax3 = sns.jointplot(data=df_all, x="theta_2", y="theta_5", hue="model", kind='kde', shade=True, multiple="layer",alpha=0.6)
    ax3.set_axis_labels('theta_1', 'theta_2')
    ax3.ax_joint.axvline(x=theta_star[0], color='black', linestyle='--', linewidth = 1)  
    ax3.ax_joint.axhline(y=theta_star[0], color='black', linestyle='--', linewidth = 1)  
    ax3.fig.suptitle(titles[2])
    ax3.fig.tight_layout()
    ax3.fig.subplots_adjust(top=0.95)
    if save_fig:
        plt.savefig(fname+'_2.png')
    #ax3.fig.get_axes()[0].legend(loc='lower left')
    
    #sns.jointplot(data=df_all, x="theta_1", y="theta_4", hue="model", kind='kde')
    #sns.jointplot(data=df_all, x="theta_2", y="theta_5", hue="model", kind='kde')
    
#    fig, ax_array = plt.subplots(1, n_cont, figsize=(10,5))
#    titles = ['{} % of outliers'.format(i*5) for i in range(n_cont)]
#    
#    for i,ax in enumerate(ax_array):
#        ax = sns.kdeplot(data=df_all, x="theta_{}".format(i%3), y="theta_{}".format((i%3)+3), hue="model",ax=ax)
#        ax.set_xlabel('theta_1')
#        ax.set_ylabel('theta_2')
#        ax.set_title(titles[i]) #i%3
#        ax.axvline(theta_star[0], color='black')  
#        ax.axhline(theta_star[0], color='black')
#    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#    if save_fig == True:
#        fig.savefig('posterior_marginal_plot_2d_gauss_biv.png')
        
    return ax1, ax2, ax3
        
        
def plot_gnk(B,thetas_wabc, thetas_mmd, theta_star, n_cont, fname, save_fig=False):
    compare = 2
    p = len(theta_star)
    theta_data = np.zeros((p,n_cont,compare*B))
    for i in range(p):
        for j in np.arange(n_cont):
            theta_data[i,j,:] = np.concatenate((thetas_wabc[i,j,:],thetas_mmd[i,j,:]))
            
    model_name_data = np.concatenate((['WABC']*B,['NPL-MMD']*B))
    columns = []
    for i in range(n_cont*p):
         columns.append('theta_{}'.format(i))  
    columns.append('model')
    theta_data = np.reshape(theta_data,(n_cont*p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(n_cont*p) }) 
    df_all['model'] =  model_name_data
    titles = ['{} % of outliers'.format(i*5) for i in range(n_cont)]
    
    axes = []
    
#    for i in range(n_cont):
#        ax = sns.jointplot(data=df_all, x="theta_{}".format(i%n_cont), y="theta_{}".format(i%n_cont+n_cont), hue="model", kind='kde', shade=True, multiple="layer",alpha=0.6)
#        ax.set_axis_labels('a', 'b')
#        ax.ax_joint.axvline(x=theta_star[0], color='black', linestyle='--', linewidth = 1)  
#        ax.ax_joint.axhline(y=theta_star[1], color='black', linestyle='--', linewidth = 1)  
#        ax.fig.suptitle(titles[i])
#        ax.fig.tight_layout()
#        ax.fig.subplots_adjust(top=0.95)
#        axes.append(ax)
#        if save_fig:
#            plt.savefig('gnk_{}_theta12.png'.format(i))
            
    for i in range(n_cont):
        ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont), y="theta_{}".format(i%n_cont+n_cont), hue="model")
        if i == 0:
            ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6)
        else:
            ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6, legend=False)
        ax.set_axis_labels('a', 'b', fontsize=18)
        ax.ax_joint.axvline(x=theta_star[0], color='black', linestyle='--', linewidth = 1)  
        ax.ax_joint.axhline(y=theta_star[1], color='black', linestyle='--', linewidth = 1)  
        ax.fig.suptitle(titles[i])
        ax.fig.tight_layout()
        ax.fig.subplots_adjust(top=0.95)
        axes.append(ax)
        if save_fig:
            plt.savefig(fname+'_{}_theta12.png'.format(i))
            
    for i in range(n_cont):
        ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont+2*n_cont), y="theta_{}".format(i%n_cont+3*n_cont), hue="model")
        ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6, legend=False)
        ax.set_axis_labels('g', 'log(k)', fontsize=18)
        ax.ax_joint.axvline(x=theta_star[2], color='black', linestyle='--', linewidth = 1)  
        ax.ax_joint.axhline(y=theta_star[3], color='black', linestyle='--', linewidth = 1)  
        #ax.fig.suptitle(titles[i])
        ax.fig.tight_layout()
        ax.fig.subplots_adjust(top=0.95)
        axes.append(ax)
        if save_fig:
            plt.savefig(fname+'_{}_theta34.png'.format(i))
    
    return axes


def plot_posterior_marg_ts(B,thetas_wabc, thetas_npl_mmd, theta_star, fname, save_fig=False):
    """Plot the posterior marginal densities obtained from the parameter sample"""
    
    compare = 2
    p = len(theta_star)
    theta_data = np.zeros((p,compare*B))
    for i in range(p):
        theta_data[i,:] = np.concatenate((thetas_npl_mmd[i,0,:],thetas_wabc[i,0,:]))
            
    model_name_data = np.concatenate((['NPL-MMD']*B,['WABC']*B))

    columns = []
    for i in range(p):
         columns.append('theta_{}'.format(i))  
    columns.append('model')
    theta_data = np.reshape(theta_data,(p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(p) }) 
    df_all['model'] =  model_name_data
    fig, ax_array = plt.subplots(3,3 , figsize=(10,10))  #(20,8)
    fig.delaxes(ax_array[2,1])
    fig.delaxes(ax_array[2,2])
    #titles = ['{} % of outliers'.format((i)*5) for i in range(n_cont)]
    colors = ['#0072B2', '#D55E00', '#009E73'] 
    xlabel = [r'$\alpha_1$', r'$\alpha_2$', r'$log(\beta_1)$', r'$\beta_2$', r'$\mu$', r'$log(\sigma)$', r'$\gamma$']
    
    for ax, i in zip(ax_array.flatten(), range(0, p)):
        if i == 0:
            ax = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="model", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False)
        else:
            ax = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="model", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False, legend=False)
        ax.set_xlabel(xlabel[i])
        #ax.set_title(titles[i%6]) #i%3
        ax.axvline(theta_star[i], color='black')  #prepei na ftiaxw ta indexes  #
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if save_fig == True:
        fig.savefig(fname)
      
    return fig, ax_array


        
    