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

# Density plots 
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
            ax.set_title(titles[i%3], fontsize=15) #i%3
        ax.axvline(theta_star[math.floor(i/n_cont)], color='black', linestyle='--', linewidth = 1)  #prepei na ftiaxw ta indexes  #
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
            theta_data[i,j,:] = np.concatenate((thetas_mmd[i,j,:],thetas_wabc[i,j,:],thetas_wll[i,j,:]))
            
    model_name_data = np.concatenate((['NPL-MMD']*B,['WABC']*B,['NPL-WLL']*B))
    columns = []
    for i in range(n_cont*p):
         columns.append('theta_{}'.format(i))  
    columns.append('method')
    theta_data = np.reshape(theta_data,(n_cont*p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(n_cont*p) }) 
    df_all['method'] =  model_name_data
    titles = ['{} % of outliers'.format(i*5) for i in range(n_cont)]
    
    ax1 = sns.jointplot(data=df_all, x="theta_0", y="theta_3", hue="method", kind='kde', shade=True, multiple="layer",alpha=0.6)
    ax1.set_axis_labels('theta_1', 'theta_2')
    ax1.ax_joint.axvline(x=theta_star[0], color='black', linestyle='--', linewidth = 1)  
    ax1.ax_joint.axhline(y=theta_star[0], color='black', linestyle='--', linewidth = 1)  
    ax1.fig.suptitle(titles[0])
    ax1.fig.tight_layout()
    ax1.fig.subplots_adjust(top=0.95)
    if save_fig:
        plt.savefig(fname+'_0.png')
    
    ax2 = sns.jointplot(data=df_all, x="theta_1", y="theta_4", hue="method", kind='kde', shade=True, multiple="layer",alpha=0.6)
    ax2.set_axis_labels('theta_1', 'theta_2')
    ax2.ax_joint.axvline(x=theta_star[0], color='black', linestyle='--', linewidth = 1)  
    ax2.ax_joint.axhline(y=theta_star[0], color='black', linestyle='--', linewidth = 1)  
    ax2.fig.suptitle(titles[1])
    ax2.fig.tight_layout()
    ax2.fig.subplots_adjust(top=0.95)
    if save_fig:
        plt.savefig(fname+'_1.png')
    
    ax3 = sns.jointplot(data=df_all, x="theta_2", y="theta_5", hue="method", kind='kde', shade=True, multiple="layer",alpha=0.6)
    ax3.set_axis_labels('theta_1', 'theta_2')
    ax3.ax_joint.axvline(x=theta_star[0], color='black', linestyle='--', linewidth = 1)  
    ax3.ax_joint.axhline(y=theta_star[0], color='black', linestyle='--', linewidth = 1)  
    ax3.fig.suptitle(titles[2])
    ax3.fig.tight_layout()
    ax3.fig.subplots_adjust(top=0.95)
    if save_fig:
        plt.savefig(fname+'_2.png')
        
    return ax1, ax2, ax3
        
        
def plot_gnk(B,thetas_wabc, thetas_mmd, theta_star, n_cont, fname, save_fig=False):
    compare = 2
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
    compare = 3
    p = len(theta_star)
    theta_data = np.zeros((p,n_cont,compare*B))
    for i in range(p):
        for j in np.arange(n_cont):
            theta_data[i,j,:] = np.concatenate((thetas_mmd[i,j,:],thetas_wabc[i,j,:], thetas_wll[i,j,:])) #thetas_was[i,j,:]
            
    model_name_data = np.concatenate((['NPL-MMD']*B, ['WABC']*B, ['NPL-WLL']*B)) #['NPL-WAS']*B
    columns = []
    for i in range(n_cont*p):
         columns.append('theta_{}'.format(i))  
    columns.append('method')
    theta_data = np.reshape(theta_data,(n_cont*p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(n_cont*p) }) 
    df_all['method'] =  model_name_data
    titles = ['{} % of outliers'.format(i*5) for i in range(n_cont)]
    
    axes = []
    
    colors = ['#0072B2', '#D55E00', '#009E73'] #,'#e31a1c']  
    order = [ 'NPL-MMD', 'WABC', 'NPL-WLL'] #'NPL-WAS',
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
        #ax.fig.suptitle(titles[i])
        ax.fig.tight_layout()
        ax.fig.subplots_adjust(top=0.95)
        axes.append(ax)
        if save_fig:
            plt.savefig(fname+'_{}_theta34.png'.format(i))
    
    return axes
    
from matplotlib.gridspec import GridSpec
def plot_grid(B,thetas_wabc, thetas_npl_mmd, theta_star, y, fname, save_fig=False):
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

    n = 1 # number of double-rows
    m = 4 # number of columns

    t = 0.9 # 1-t == top space 
    b = 0.1 # bottom space      (both in figure coordinates)

    msp = 0.1 # minor spacing
    sp = 0.6  # major spacing

    offs=(1+msp)*(t-b)/(2*n+n*msp+(n-1)*sp) # grid offset
    hspace = sp+msp+1 #height space per grid
    
    gs = GridSpec(3,4, hspace=0.1, height_ratios=[1, 1, 2])

    fig = plt.figure(figsize=(20,12))
    ax_array = []
    for i in range(12):
        ax_array.append(fig.add_subplot(gs[i]))
       
    first_ax_set = [ax_array[i] for i in [0,1,2,3]]
    snd_ax_set = [ax_array[i] for i in [4,5,6,7]]

    colors = ['#0072B2', '#D55E00', '#009E73'] 
    xlabel = [r'$\alpha_1$', r'$\alpha_2$', r'$\beta_1$', r'$\sigma$', r'$\beta_2$', r'$\mu$', r'$\gamma$']
    
    for ax1,ax2, i in zip(first_ax_set, snd_ax_set, range(0, 4)):
        if i == 0:
            ax1 = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax1, alpha=0.6,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False)
            ax2 = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2,  ax=ax2, alpha=0.6,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False)
            ax1.set_ylim(4, 6)
            ax2.set_ylim(0, 2)
            ax1.get_xaxis().set_visible(False)
            ax1.axvline(theta_star[i], color='black', linestyle='--', linewidth = 1)
            ax2.axvline(theta_star[i], color='black', linestyle='--', linewidth = 1)
            ax1.set_title(xlabel[i],fontsize=18)
            ax1.set_ylabel("")
            ax2.set_ylabel("")
            ax2.get_legend().remove()
            ax1.set_yticklabels(ax1.get_yticks(), size = 15)
            ax2.set_yticklabels(ax2.get_yticks(), size = 15)
            ax2.set_xticklabels(ax2.get_xticks(), size = 15)
            ax1.legend_._set_loc(9)
            plt.setp(ax1.get_legend().get_texts(), fontsize='18') 
            plt.setp(ax1.get_legend().get_title(), fontsize='18') 
            ylabels2 = ['{:}'.format(x) for x in ax2.get_yticks()]
            ax2.set_yticklabels(ylabels2, size = 15)
            ylabels1 = ['{:}'.format(x) for x in ax1.get_yticks()]
            ax1.set_yticklabels(ylabels1, size = 15)
            xlabelst = ['{:}'.format(int(x)) for x in ax2.get_xticks()]
            ax2.set_xticklabels(xlabelst, size = 15)
            
            d = .02  # how big to make the diagonal lines in axes coordinates
            # arguments to pass to plot, just so we don't keep repeating them
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        else:
            if i == 1:
                ax1.set_ylim(0.3, 0.5)
                ax2.set_ylim(0, 0.03)
                ax1 = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax1, alpha=0.6,hue="method",multiple="layer", palette=colors[0:compare], shade=True, common_norm=True,common_grid=True, legend=False)
                ax2 = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax2, alpha=0.6,hue="method",multiple="layer", palette=colors[0:compare], shade=True, common_norm=True,common_grid=True, legend=False)
                ax1.axvline(theta_star[i], color='black',  linestyle='--', linewidth = 1)  
                ax2.axvline(theta_star[i], color='black',  linestyle='--', linewidth = 1)
                ylabels2 = ['{:}'.format(x) for x in ax2.get_yticks()]
                ax2.set_yticklabels(ylabels2, size = 15)
                ylabels1 = ['{:}'.format(x) for x in ax1.get_yticks()]
                ax1.set_yticklabels(ylabels1, size = 15)
                xlabelst = ['{:}'.format(int(x)) for x in ax2.get_xticks()]
                ax2.set_xticklabels(xlabelst, size = 15)
                
                
            elif i == 2:
                ax1.set_ylim(3.5, 6)
                ax2.set_ylim(0, 0.17)
                ax1 = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax1, alpha=0.6,hue="method",multiple="layer", palette=colors[0:compare], shade=True, common_norm=True,common_grid=True, legend=False)
                ax2 = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax2, alpha=0.6,hue="method",multiple="layer", palette=colors[0:compare], shade=True, common_norm=True,common_grid=True, legend=False)
                ax1.axvline(theta_star[i], color='black',  linestyle='--', linewidth = 1)  
                ax2.axvline(theta_star[i], color='black',  linestyle='--', linewidth = 1)
                ylabels2 = ['{:,.2f}'.format(x) for x in ax2.get_yticks()]
                ax2.set_yticklabels(ylabels2, size = 15)
                ylabels1 = ['{:,.2f}'.format(x) for x in ax1.get_yticks()]
                ax1.set_yticklabels(ylabels1, size = 15)
                xlabelst = ['{:}'.format(int(x)) for x in ax2.get_xticks()]
                ax2.set_xticklabels(xlabelst, size = 15)
            elif i == 3:
                ax1 = sns.kdeplot(data=df_all, x="theta_{}".format(i+2), linestyle="-", linewidth = 2, ax=ax1, alpha=0.6,hue="method",multiple="layer", palette=colors[0:compare], shade=True, common_norm=True,common_grid=True, legend=False)
                ax2 = sns.kdeplot(data=df_all, x="theta_{}".format(i+2), linestyle="-", linewidth = 2, ax=ax2, alpha=0.6,hue="method",multiple="layer", palette=colors[0:compare], shade=True, common_norm=True,common_grid=True, legend=False)
                ax1.set_ylim(70, 90)
                ax2.set_ylim(0, 9)
                ax1.axvline(theta_star[i+2], color='black',  linestyle='--', linewidth = 1)  
                ax2.axvline(theta_star[i+2], color='black',  linestyle='--', linewidth = 1)
                ylabels2 = ['{:}'.format(int(x)) for x in ax2.get_yticks()]
                ax2.set_yticklabels(ylabels2, size = 15)
                ylabels1 = ['{:}'.format(int(x)) for x in ax1.get_yticks()]
                ax1.set_yticklabels(ylabels1, size = 15)
                xlabelst = ['{:,.2f}'.format(x) for x in ax2.get_xticks()]
                ax2.set_xticklabels(xlabelst, size = 15)
            
            ax1.get_xaxis().set_visible(False)
            ax1.set_title(xlabel[i],fontsize=18)
            ax1.set_ylabel("")
            ax2.set_ylabel("")

            d = .02  
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
            
    for ax,i in zip([ax_array[j] for j in [8,9,10,11]],range(4)):
        if i == 3:
            df1=pd.DataFrame.from_dict({'value': y, ' ': 'Data without noise'})
            ax = sns.histplot(data=df1, x='value', hue=' ', multiple='dodge',ax=ax, palette=['#0072B2'])
            ax.set_ylabel("") 
            ax.set_xlabel('x',fontsize=18)
            plt.setp(ax.get_legend().get_texts(), fontsize='18')    
            ax.set_yticklabels(ax.get_yticks(), size = 15) 
            xlabelst = ['{:}'.format(int(x)) for x in ax.get_xticks()]
            ax.set_xticklabels(xlabelst, size = 15) 
            ylabels = ['{:}'.format(int(x)) for x in ax.get_yticks()]
            ax.set_yticklabels(ylabels, size = 15)                                                                         
        elif i == 0:                                                                                    
            ax = sns.kdeplot(data=df_all, x="theta_{}".format(3), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False, legend=False)
            ax.set_xlabel(xlabel[4],fontsize=18)
            ax.set_ylabel("")
            ax.axvline(theta_star[3], color='black',  linestyle='--', linewidth = 1)
            ylabels = ['{:,.2f}'.format(x) for x in ax.get_yticks()]
            ax.set_yticklabels(ylabels, size = 15)
            ax.set_xticklabels(ax.get_xticks(), size = 15) 
        elif i == 1:
            ax = sns.kdeplot(data=df_all, x="theta_{}".format(4), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False, legend=False)
            ax.set_xlabel(xlabel[5],fontsize=18)
            ax.set_ylabel("")
            ax.axvline(theta_star[4], color='black',  linestyle='--', linewidth = 1)
            ylabels = ['{:,.2f}'.format(x) for x in ax.get_yticks()]
            ax.set_yticklabels(ylabels, size = 15)
            xlabelst = ['{:}'.format(int(x)) for x in ax.get_xticks()]
            ax.set_xticklabels(xlabelst, size = 15) 
        else:
            ax = sns.kdeplot(data=df_all, x="theta_{}".format(6), linestyle="-", linewidth = 2, ax=ax, alpha=0.3,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False, legend=False)
            ax.set_xlabel(xlabel[6],fontsize=18)
            ax.axvline(theta_star[6], color='black',  linestyle='--', linewidth = 1)
            ax.set_ylabel("")
            ylabels = ['{:}'.format(int(x)) for x in ax.get_yticks()]
            xlabelst = ['{:,.1f}'.format(x) for x in ax.get_xticks()]
            ax.set_yticklabels(ylabels, size = 15) 
            ax.set_xticklabels(xlabelst, size = 15) 
        
    fig.tight_layout()
    if save_fig == True:
        fig.savefig(fname)
    return fig, ax_array

def plot_grid2(B,thetas_wabc, thetas_npl_mmd, theta_star, y, fname, save_fig=False):
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

    ax_array = []
    colors = ['#0072B2', '#D55E00', '#009E73'] 
    xlabel = [r'$\alpha_1$', r'$\alpha_2$', r'$\beta_1$', r'$\beta_2$', r'$\mu$', r'$\sigma$', r'$\gamma$']
    
    for i in range(0, 8):
        if i == 0:
            f, (ax1, ax2) = plt.subplots(ncols=1, nrows=2,sharex=True)
            ax1 = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax1, alpha=0.6,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False)
            ax2 = sns.kdeplot(data=df_all, x="theta_{}".format(i), linestyle="-", linewidth = 2, ax=ax2, alpha=0.6,hue="method", multiple="layer", palette=colors[0:compare], shade=True, common_norm=False)
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
            
            d = .02  # how big to make the diagonal lines in axes coordinates
            # arguments to pass to plot, just so we don't keep repeating them
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
            ax_array.append((ax1,ax2))
    return ax_array

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

def plot_gauss_4d_2(B,thetas_mmd, thetas_mabc, theta_star, n_cont, fname, save_fig=False):
    compare = 2
    p = len(theta_star)
    theta_data = np.zeros((p,n_cont,compare*B))
    for i in range(p):
        for j in np.arange(n_cont):
            theta_data[i,j,:] = np.concatenate((thetas_mmd[i,j,:],thetas_mabc[i,j,:]))
            
    model_name_data = np.concatenate((['NPL-MMD']*B, ['MMD-ABC']*B))
    columns = []
    for i in range(n_cont*p):
         columns.append('theta_{}'.format(i))  
    columns.append('method')
    theta_data = np.reshape(theta_data,(n_cont*p,compare*B))
    df_all = pd.DataFrame({'theta_{}'.format(k): theta_data[k,:] for k in range(n_cont*p) }) 
    df_all['method'] =  model_name_data
    titles = ['{} % of outliers'.format(i*5) for i in range(n_cont)]
    axes=[]
    colors = ['#0072B2', '#D55E00']      
    for i in range(n_cont):
        
        if i == 0:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont), y="theta_{}".format(i%n_cont+n_cont), hue="method", palette=colors)#, hue_order=['NPL-WAS','NPL-MMD'])#, ylim=(-4,3))
            ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6)
        elif i == 1:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont), y="theta_{}".format(i%n_cont+n_cont), hue="method", palette=colors, xlim=(0,2), ylim=(-3,3))#, hue_order=['NPL-WAS','NPL-MMD']) #
            ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6, legend=False)
        else:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont), y="theta_{}".format(i%n_cont+n_cont), hue="method", palette=colors, xlim=(0,1.4), ylim=(-3,3))#, hue_order=['NPL-WAS','NPL-MMD']) #)
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
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont+2*n_cont), y="theta_{}".format(i%n_cont+3*n_cont), hue="method",  palette=colors, xlim=(-1, 4)) #, ylim=(-0.5,4))#,hue_order=['NPL-WAS','NPL-MMD'])#)
        elif i == 1:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont+2*n_cont), y="theta_{}".format(i%n_cont+3*n_cont), hue="method", palette=colors, xlim=(-2, 4), ylim=(-1,5))#, hue_order=['NPL-WAS','NPL-MMD'])#)
        else:
            ax = sns.JointGrid(data=df_all, x="theta_{}".format(i%n_cont+2*n_cont), y="theta_{}".format(i%n_cont+3*n_cont), hue="method", palette=colors, xlim=(-1, 5), ylim=(-1,6))#, hue_order=['NPL-WAS','NPL-MMD'])#)
        ax.plot(sns.kdeplot, sns.kdeplot, shade=True, multiple="layer",alpha=0.6, legend=False)
        ax.set_axis_labels(r'$\exp(\theta_3)$', r'$\exp(\theta_4)$', fontsize=18)
        ax.ax_joint.axvline(x=theta_star[2], color='black', linestyle='--', linewidth = 1)  
        ax.ax_joint.axhline(y=theta_star[3], color='black', linestyle='--', linewidth = 1)  
        #ax.fig.suptitle(titles[i])
        ax.fig.tight_layout()
        ax.fig.subplots_adjust(top=0.95)
        axes.append(ax)
        if save_fig:
            plt.savefig(fname+'_{}_theta34.png'.format(i))
    
    return axes

class SeabornFig2Grid():

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




