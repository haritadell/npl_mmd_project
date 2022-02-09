### This folder contains all necessary files to perform inference under the Nonparametric Learning framework (NPL) using the Posterior Bootstrap. 
- `utlis.py` contains functions used to sample from the contaminated models, kernel and MMD approximation functions.
- `models.py` contains model classes for each of the Gaussian location model, G-and-k distribution and Toggle Switch model.
- `NPL.py` contains the necessary code to perform inference for any model in `models.py` using the Posterior Bootstrap for the case α = 0 in the DP prior.
- `NPL_prior.py` contains the necessary code to perform inference for any model in `models.py` using the Posterior Bootstrap. 
-  `plot_functions` contains funcctions for the plots used in the paper.
-  `compute_mses.py` computes MSEs over multiple runs. 
