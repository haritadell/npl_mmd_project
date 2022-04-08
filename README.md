# Robust Bayesian Inference for Simulator-based models via the MMD Posterior Bootstrap
This repository contains all code needed to recreate the results in the paper "Robust Bayesian Inference for Simulator-based models via the MMD Posterior Bootstrap" by 
Charita Dellaporta, Jeremias Knoblauch, Theodoros Damoulas and François-Xavier Briol. The code for the MMD posterior bootstrap is written in Python. Our method uses [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) which is 
compatible with GPU usage, hence we provide both the files needed to run experiments locally and an .ipynb file compatible for use with Google Colaboratory. 
We use the R code (`ABC_Rcode` folder) provided by Bernton, Jacob, Gerbert & Robert (2019) [here](https://github.com/pierrejacob/winference/tree/master/inst/reproduceabc) to compare against ABC with the Wasserstein distance as well as the code by Pacchiardi & Dutta (2021) [here](https://github.com/LoryPack/GenBayes_LikelihoodFree_ScoringRules) to compare against MMD-Bayes with the kernel score (Appendix, Figure 10).
A more detailed README file can be found inside the `src` folder.

## Requirements 
- Python == 3.7.*
- Numpy == 1.19.5
- Jax == 0.2.13
- SciPy == 1.4.1
- Seaborn == 0.11.2

## Reproducing experiments 
- The folder `data` contains all the datasets used for the experiments.
- All scripts have paths to folders named `data`, `results` and  `plots`. You can create such folders or set your own paths. 
- To reproduce the experiments for the NPL based methods (i.e. NPL-MMD, NPL-WAS and NPL-WLL) locally, run `run_gaussian.py`, `run_gandk.py` and `run_togswitch.py` by setting the relevant data/results/plots paths 
directly after imports and indicating whether you want to generate new datasets or use the ones used in the paper (located in `data` folder). 
The files will run, save and plot the results to the relevant paths. Alternatively, the notebook `Experiments_notebook.ipynb` is optimised for use with [Google colab](https://colab.research.google.com/).
The notebook mounts your google drive and calls all the relevant py scripts so you need to import the `src` folder in your google drive. 
- To run the experiments for the same datasets using the Wasserstein-ABC method run the `run_wabc_experiments.R` file which is entirely based on the code provided
by the authors in Bernton, Jacob, Gerbert & Robert (2019). 

## References
- Bernton, E., Jacob, P.E., Gerber, M. and Robert, C.P., 2019. Approximate Bayesian computation with the Wasserstein distance. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 81(2), pp.235-269
- Pacchiardi, L. and Dutta, R., 2021. Generalized Bayesian likelihood-free inference using scoring rules estimators. arXiv preprint arXiv:2104.03889

