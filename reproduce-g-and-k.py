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
# 2) Indicate whether you want a new dataset or to load existing one 
# 3) Experiments are run for multiple runs - index which run you want plots for

# Set paths
data_path = "./data/G_and_k_model/"
results_path = "./results/G_and_k_model/new/"

# Set to True to generate and save new datasets or False to load saved datasets
sample_data_bool = False

# Set model 
model_name = 'gandk' 
n = 2**11 # number of observations
d = 1 # dimension of data
theta_star = np.array([3,1,1,-np.log(2)]) # true parameter value 
outl = 3 # number of different percentages of outliers to run for
m = 2**9 # number of samples within NPL
l = 0.15  # kernel lengthscale
p = 4   # number of unknown parameters
B = 5 # number of posterior samples
model = models.g_and_k_model(m,d)
R = 1 # number of independent runs
s=1 # std of Gaussian data

#######################################################################
###### Replace datasets here with your own datasets if you want  
### datasets needs to be of dimensions R x outl x n where R is number of datasets, outl is number of outlier settings and n is number of observations

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
######################################################################    

# Obtain and save results 
if __name__=='__main__':
    times = []
    # summary_stats = np.zeros((R,outl, p, 4)) # collect mean, median, mode, st.dev for each bootstrap sample
    for j in range(R):
        # print("-----Run ", j)
        for n_cont in range(outl):
            # print("-----Running for", n_cont*5, "% of outliers-----")
            X =datasets[j,n_cont,:].reshape((n,1))
            npl = NPL.npl(X,B,m,p,l, model = model, model_name = model_name)
            t0 = time.time()
            npl.draw_samples()
            t1 = time.time()
            total = t1- t0
            times.append(total)
            sample = npl.sample
            np.savetxt(results_path+'NPL_MMD/thetas_mmd_outl_{}_run_{}.txt'.format(n_cont,j), sample)
            
    np.savetxt(results_path+'NPL_MMD/cpu_times.txt', times) 
