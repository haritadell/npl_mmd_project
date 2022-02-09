#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:31:00 2021

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()

def plot_posterior_marginals(B,thetas_wabc, thetas_npl_mmd, thetas_npl_wll, thetas_npl_was, theta_star, n_cont, fname, gaussian=False, save_fig=False):
    """Plot the posterior marginal densities obtained from the parameter sample"""
    if gaussian==True:
        compare = 4
        p = len(theta_star)
        theta_data = np.zeros((p,n_cont,compare*B))
        for i in range(p):
            for j in np.arange(n_cont):
                theta_data[i,j,:] = np.concatenate((thetas_npl_mmd[i,j,:],thetas_wabc[i,j,:],thetas_npl_wll[i,j,:],thetas_npl_was[i,j,:]))
            
        model_name_data = np.concatenate((['NPL-MMD']*B,['WABC']*B,['NPL-WLL']*B,['NPL-WAS']*B))
    else:
        compare = 2
        p = len(theta_star)
        theta_data = np.zeros((p,n_cont,compare*B))
        for i in range(p):
            for j in np.arange(n_cont):
                theta_data[i,j,:] = np.concatenate((thetas_npl_mmd[i,j,:],thetas_wabc[i,j,:]))
            
        model_name_data = np.concatenate((['NPL-MMD']*B,['MMD-ABC']*B))

    columns = []
    for i in range(n_cont*p):
         columns.append('theta_{}'.format(i))  
    columns.append('method')
    theta_data = np.reshape(theta_data,(n_cont*p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(n_cont*p) }) 
    df_all['method'] =  model_name_data
    fig, ax_array = plt.subplots(p, n_cont, figsize=(10,8))  #(20,8)
    titles = [r'$\epsilon = {}$'.format((i)*0.05) for i in range(n_cont)]
    colors = ['#0072B2', '#D55E00', '#009E73','#dede00'] 
    
    for ax, i in zip(ax_array.flatten(), range(0, p * n_cont)):
        if i == 0:
            ax = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False)
        else:
          ax = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False, legend=False)
        ax.set_xlabel(r'$\theta_{}$'.format(math.floor(i/n_cont)+1),  fontsize=15)
        ax.set_ylabel('')
        if i <3:
            ax.set_title(titles[i%3], fontsize=15) 
        ax.axvline(theta_star[math.floor(i/n_cont)], color='black', linestyle='--', linewidth = 1)  
    fig.tight_layout()
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
        ax_array.legend(loc='best', ncol=1)
        ax_array.set_xlabel('Percentage of outliers',fontsize='x-large')
        ax_array.set_ylabel('MSE',fontsize='x-large')
        ax_array.set_title('MSE for the mean of the univariate Gaussian distribution')
    if save_fig == True:
        fig.savefig(fname)
    fig.tight_layout() 
    return fig, ax_array

def plot_gnk(B,thetas_wabc, thetas_mmd, theta_star, n_cont, fname, save_fig=False):
    """Plot function for posterior maginal distributions obtained for the G-and-k distribution.
    This function produces the individual axes which are then passed on the SeabornFig2Grid class"""
    
    compare = 2 # How many methods are we comparing 
    p = len(theta_star)
    theta_data = np.zeros((p,n_cont,compare*B))
    for i in range(p):
        for j in np.arange(n_cont):
            theta_data[i,j,:] = np.concatenate((thetas_mmd[i,j,:],thetas_wabc[i,j,:]))
            
    model_name_data = np.concatenate((['NPL-MMD']*B,['WABC']*B))
    columns = []
    for i in range(n_cont*p):
         columns.append('theta_{}'.format(i))  
    columns.append('method')
    theta_data = np.reshape(theta_data,(n_cont*p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(n_cont*p) }) 
    df_all['method'] =  model_name_data
    titles = ['{} % of outliers'.format(i*5) for i in range(n_cont)]
    
    axes = []
            
    for i in range(n_cont):
        ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont), y="theta_{}".format(i%n_cont+n_cont), hue="method")
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
        ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont+2*n_cont), y="theta_{}".format(i%n_cont+3*n_cont), hue="method")
        ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6, legend=False)
        ax.set_axis_labels('g', 'log(k)', fontsize=18)
        ax.ax_joint.axvline(x=theta_star[2], color='black', linestyle='--', linewidth = 1)  
        ax.ax_joint.axhline(y=theta_star[3], color='black', linestyle='--', linewidth = 1)  
        ax.fig.tight_layout()
        ax.fig.subplots_adjust(top=0.95)
        axes.append(ax)
        if save_fig:
            plt.savefig(fname+'_{}_theta34.png'.format(i))
    
    return axes


def plot_posterior_marg_ts(B,thetas_wabc, thetas_npl_mmd, theta_star, y, fname, save_fig=False):
    """Plot the posterior marginal densities obtained for the Toggle switch model"""
    
    compare = 2
    p = len(theta_star)
    theta_data = np.zeros((p,compare*B))
    for i in range(p):
        theta_data[i,:] = np.concatenate((thetas_npl_mmd[i,:],thetas_wabc[i,:]))
            
    model_name_data = np.concatenate((['NPL-MMD']*B,['WABC']*B))

    columns = []
    for i in range(p):
         columns.append('theta_{}'.format(i))  
    columns.append('method')
    theta_data = np.reshape(theta_data,(p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(p) }) 
    df_all['method'] =  model_name_data
    fig, ax_array = plt.subplots(6,3 , figsize=(10,10))  #(20,8)
    fig.delaxes(ax_array[4,2])
    fig.delaxes(ax_array[5,2])
    first_ax_set = [ax_array.flatten()[i] for i in [0,1,2,6,7,8,12,13]]
    snd_ax_set = [ax_array.flatten()[i] for i in [3,4,5,9,10,11,15,16]]

    colors = ['#0072B2', '#D55E00', '#009E73'] 
    xlabel = [r'$\alpha_1$', r'$\alpha_2$', r'$\beta_1$', r'$\beta_2$', r'$\mu$', r'$\sigma$', r'$\gamma$']
    
    for ax1,ax2, i in zip(first_ax_set, snd_ax_set, range(0, 8)):
        if i == 0:
            ax1 = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax1, alpha=0.6,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False)
            ax2 = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2,  ax=ax2, alpha=0.6,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False)
            ax1.set_ylim(4, 6)
            ax2.set_ylim(0, 2)
            ax1.get_xaxis().set_visible(False)
            ax1.axvline(theta_star[i], color='black', linestyle='--', linewidth = 1)
            ax2.axvline(theta_star[i], color='black', linestyle='--', linewidth = 1)
            ax2.set_xlabel(xlabel[i],fontsize=18)
            ax1.set_ylabel("")
            ax2.set_ylabel("")
            ax2.get_legend().remove()
            ax1.legend_._set_loc(9)
            sns.set_style("darkgrid")
            
        elif i == 7:
            df1=pd.DataFrame.from_dict({'value': y, ' ': 'Data without noise'})
            ax1 = sns.histplot(data=df1, x='value', hue=' ', multiple='dodge',ax=ax1, palette=['#0072B2'])
            ax2 = sns.histplot(data=df1, x='value', hue=' ', multiple='dodge',ax=ax2, palette=['#0072B2'])
            ax2.set_ylim(0, 400)
            ax1.set_ylim(600, 1100)
            ax1.get_xaxis().set_visible(False)
            ax1.set_ylabel("")
            ax2.set_ylabel("")
            ax2.get_legend().remove()
            ax1.legend_._set_loc(9)
            sns.set_style("darkgrid")
        else:
            ax1 = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax1, alpha=0.6,hue="method",multiple="layer", palette=colors[0:compare], shade=True, common_norm=True,common_grid=True, legend=False)
            ax2 = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax2, alpha=0.6,hue="method",multiple="layer", palette=colors[0:compare], shade=True, common_norm=True,common_grid=True, legend=False)
            if i == 1:
                ax1.set_ylim(0.3, 0.5)
                ax2.set_ylim(0, 0.03)
            elif i == 2:
                ax1.set_ylim(3.5, 5)
                ax2.set_ylim(0, 0.17)
            elif i==3:
                ax1.set_ylim(0.075, 0.176)
                ax2.set_ylim(0, 0.075)
            elif i ==4:
                ax1.set_ylim(0.06, 0.09)
                ax2.set_ylim(0, 0.04)
            elif i == 5:
                ax1.set_ylim(70, 90)
                ax2.set_ylim(0, 9)
            elif i == 6:
                ax1.set_ylim(4, 8.2)
                ax2.set_ylim(0, 4)
            ax1.get_xaxis().set_visible(False)
            ax2.set_xlabel(xlabel[i],fontsize=18)
            ax1.set_ylabel("")
            ax2.set_ylabel("")
            ax1.axvline(theta_star[i], color='black',  linestyle='--', linewidth = 1)  
            ax2.axvline(theta_star[i], color='black',  linestyle='--', linewidth = 1)
            sns.set_style("darkgrid")

    plt.subplots_adjust(hspace=0.2)

    if save_fig == True:
        fig.savefig(fname)
      
    return fig, ax_array


def plot_gauss_4d(B,thetas_wabc, thetas_mmd, thetas_wll, thetas_was, theta_star, n_cont, fname, save_fig=False):
    """Plots posterior marginal densities for the Gaussian location model with d=4 in bivariate plots.
    This function produces the individual axes which are then passed on the SeabornFig2Grid class"""
    
    compare = 3 # Compare three methods
    p = len(theta_star)
    theta_data = np.zeros((p,n_cont,compare*B))
    for i in range(p):
        for j in np.arange(n_cont):
            theta_data[i,j,:] = np.concatenate((thetas_mmd[i,j,:],thetas_wabc[i,j,:], thetas_wll[i,j,:])) 
            
    model_name_data = np.concatenate((['NPL-MMD']*B, ['WABC']*B, ['NPL-WLL']*B)) 
    columns = []
    for i in range(n_cont*p):
         columns.append('theta_{}'.format(i))  
    columns.append('method')
    theta_data = np.reshape(theta_data,(n_cont*p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(n_cont*p) }) 
    df_all['method'] =  model_name_data
    titles = ['{} % of outliers'.format(i*5) for i in range(n_cont)]
    
    axes = []
    
    colors = ['#0072B2', '#D55E00', '#009E73']   
    order = [ 'NPL-MMD', 'WABC', 'NPL-WLL'] 
    for i in range(n_cont):
        
        if i == 0:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont), y="theta_{}".format(i%n_cont+n_cont), hue="method", hue_order=order, palette=colors, xlim=(0.6,1.8), ylim=(-4,2.5))#
            ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6)
        elif i == 1:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont), y="theta_{}".format(i%n_cont+n_cont), hue="method", hue_order=order, palette=colors, xlim=(0.5,3), ylim=(-2.5,3))
            ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6, legend=False)
        else:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont), y="theta_{}".format(i%n_cont+n_cont), hue="method", hue_order=order, palette=colors,xlim=(0,4), ylim=(-2.5,4))
            ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6, legend=False)
        ax.set_axis_labels(r'$\theta_1$', r'$\theta_2$', fontsize=18)
        ax.ax_joint.axvline(x=theta_star[0], color='black', linestyle='--', linewidth = 1)  
        ax.ax_joint.axhline(y=theta_star[1], color='black', linestyle='--', linewidth = 1)  
        ax.fig.suptitle(titles[i])
        ax.fig.tight_layout()
        ax.fig.subplots_adjust(top=0.95)
        axes.append(ax)
        if save_fig:
            plt.savefig(fname+'_{}_theta12.png'.format(i))
            
    for i in range(n_cont):
        if i == 0:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont+2*n_cont), y="theta_{}".format(i%n_cont+3*n_cont), hue="method", palette=colors, hue_order=order,xlim=(0, 4), ylim=(-0.5,4))
        elif i == 1:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont+2*n_cont), y="theta_{}".format(i%n_cont+3*n_cont), hue="method", palette=colors, hue_order=order ,xlim=(-1, 8), ylim=(-0.5,8.5))
        else:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont+2*n_cont), y="theta_{}".format(i%n_cont+3*n_cont), hue="method", palette=colors, hue_order=order, xlim=(-1, 5), ylim=(-1.5,6))
        ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6, legend=False)
        ax.set_axis_labels(r'$\exp(\theta_3)$', r'$\exp(\theta_4)$', fontsize=18)
        ax.ax_joint.axvline(x=theta_star[2], color='black', linestyle='--', linewidth = 1)  
        ax.ax_joint.axhline(y=theta_star[3], color='black', linestyle='--', linewidth = 1)  
        ax.fig.tight_layout()
        ax.fig.subplots_adjust(top=0.95)
        axes.append(ax)
        if save_fig:
            plt.savefig(fname+'_{}_theta34.png'.format(i))
    
    return axes
    


def plot_posterior_marg_tsols(B,thetas_wabc, thetas_npl_mmd, theta_star, y, fname, save_fig=False):
    """Plot the posterior marginal densities obtained from the parameter sample"""
    
    compare = 2
    p = len(theta_star)
    theta_data = np.zeros((p,compare*B))
    for i in range(p):
        theta_data[i,:] = np.concatenate((thetas_npl_mmd[i,:],thetas_wabc[i,:]))
            
    model_name_data = np.concatenate((['NPL-MMD']*B,['WABC']*B))

    columns = []
    for i in range(p):
         columns.append('theta_{}'.format(i))  
    columns.append('model')
    theta_data = np.reshape(theta_data,(p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(p) }) 
    df_all['model'] =  model_name_data
    fig, ax_array = plt.subplots(2,4 , figsize=(13,5)) 
    colors = ['#0072B2', '#D55E00'] 
    xlabel = [r'$\alpha_1$', r'$\alpha_2$', r'$\beta_1$', r'$\beta_2$', r'$\mu$', r'$\sigma$', r'$\gamma$', 'x']
    
    for ax, i in zip(ax_array.flatten(), range(0, (p+1))):
        if i == 0:
            ax = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="model", multiple="layer", palette=colors, shade=True, common_norm=False)
            ax.set_ylim(0, 4)
            ax.legend_._set_loc(9)
            ax.axvline(theta_star[i], color='black', linestyle='--', linewidth = 1) 
            sns.set_theme()
        elif i == 7:
            df1=pd.DataFrame.from_dict({'value': y, ' ': 'Data without noise'})
            ax = sns.histplot(data=df1, x='value', hue=' ', multiple='dodge',ax=ax, palette=['#009E73'])
            ax.set_ylabel("") 
            ax.legend_._set_loc(2)
            sns.set_theme()
        else:
            ax = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="model", multiple="layer", palette=colors, shade=True, common_norm=False, legend=False)
            if i == 1:
                ax.set_ylim(0, 0.1)
                ax.set_ylabel("") 
            elif i == 2:
                ax.set_ylim(0, 0.6)
                ax.set_ylabel("") 
            elif i == 5:
                ax.set_ylim(0,25)
            elif i == 3:
                ax.set_ylabel("") 
            elif i == 6:
                ax.set_ylabel("") 
            sns.set_theme()
            ax.axvline(theta_star[i], color='black', linestyle='--', linewidth = 1) 
        ax.set_xlabel(xlabel[i], fontsize=18)
    fig.tight_layout() 
    if save_fig == True:
        fig.savefig(fname)
      
    return fig, ax_array

def plot_posterior_marginals_mmd_vs_mabc(B,thetas_mmd, thetas_mabc, theta_star, n_cont, fname, save_fig=False):
    """Plots posterior marginal densities for the Gaussian location model with d=4 in bivariate plots for 
    the NPL-MMD and the MMD-ABC. This function produces the individual axes which are then passed
    on the SeabornFig2Grid class"""
    
    compare = 2
    p = len(theta_star)
    theta_data = np.zeros((p,compare*B))
    for i in range(p):
        theta_data[i,:] = np.concatenate((thetas_mmd[i,n_cont,:],thetas_mabc[i,n_cont,:]))
        
    model_name_data = np.concatenate((['NPL-MMD']*B,['MMD-ABC']*B))

    columns = []
    for i in range(p):
         columns.append('theta_{}'.format(i))  
    columns.append('method')
    theta_data = np.reshape(theta_data,(p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(p) }) 
    df_all['method'] =  model_name_data
    fig, ax_array = plt.subplots(1, p, figsize=(8,2.5))  
    colors = ['#0072B2', '#D55E00', '#009E73','#dede00'] 
    
    for ax, i in zip(ax_array.flatten(), range(0, p)):
        if i == 3:
            ax = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False)
        else:
            ax = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False, legend=False)
        if i < 2:
            ax.set_xlabel(r'$\theta_{}$'.format(i+1),  fontsize=15)
        else:
            ax.set_xlabel(r'$exp(\theta_{})$'.format(i+1),  fontsize=15)
            
        ax.set_ylabel('')
        ax.axvline(theta_star[i], color='black', linestyle='--', linewidth = 1)  
    fig.tight_layout()
    if save_fig == True:
        fig.savefig(fname)
      
    return fig, ax_array


def plot_gauss_4d_3(B,thetas_mmd, thetas_mabc, theta_star, n_cont, fname, save_fig=False):
    """Plots posterior marginal densities for the Gaussian location model with d=4 in bivariate plots for 
    the NPL-MMD and the NPL-WAS. This function produces the individual axes which are then passed
    on the SeabornFig2Grid class"""
    
    compare = 2
    p = len(theta_star)
    theta_data = np.zeros((p,compare*B))
    for i in range(p):
        theta_data[i,:] = np.concatenate((thetas_mmd[i,n_cont,:],thetas_mabc[i,n_cont,:]))
            
    model_name_data = np.concatenate((['NPL-MMD']*B, ['NPL-WAS']*B))
    columns = []
    for i in range(p):
         columns.append('theta_{}'.format(i))  
    columns.append('method')
    theta_data = np.reshape(theta_data,(p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(p) }) 
    df_all['method'] =  model_name_data
    axes=[]
    colors = ['#e41a1c', '#0072B2'] 
    for i in range(2):
        
        if i == 0:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i), y="theta_{}".format(i+1), hue="method", palette=colors, hue_order=['NPL-WAS','NPL-MMD']) #, xlim=(0.6,2),ylim=(-2,2.2))##, ylim=(-4,3))
            ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6, legend=False)
            ax.set_axis_labels(r'$\theta_1$', r'$\theta_2$', fontsize=18)
            ax.ax_joint.axvline(x=theta_star[0], color='black', linestyle='--', linewidth = 1)  
            ax.ax_joint.axhline(y=theta_star[1], color='black', linestyle='--', linewidth = 1) 
        elif i == 1:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i+1), y="theta_{}".format(i+2), hue="method", palette=colors,   hue_order=['NPL-WAS','NPL-MMD']) #, xlim=(0.5,6), ylim=(0,6)) #
            ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6)
            ax.set_axis_labels(r'$\exp(\theta_3)$', r'$\exp(\theta_4)$', fontsize=18)
            ax.ax_joint.axvline(x=theta_star[2], color='black', linestyle='--', linewidth = 1)  
            ax.ax_joint.axhline(y=theta_star[3], color='black', linestyle='--', linewidth = 1)  
        #ax.fig.suptitle(titles[i])
        ax.fig.tight_layout()
        ax.fig.subplots_adjust(top=0.95)
        axes.append(ax)
        if save_fig:
            plt.savefig(fname+'_{}_theta12.png'.format(i))
    
    return axes


class SeabornFig2Grid():
    """Class taken from https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot
    which allows to put multiple JointPlots into figure with multiple subplots"""
    
    def __init__(self, seaborngrid, fig,  subplot_spec, title, add_title=False):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        self.title = title
        self.add_title = add_title
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()
        

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1], add=True)
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs, add = False):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)
        if add == True:
            if self.add_title == True:
                ax.set_title(self.title, fontsize=18)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())




