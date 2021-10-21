library(winference)
registerDoParallel(cores = detectCores())
rm(list = ls())
setmytheme()
set.seed(11)

prefix = ""

target <- get_toggleswitch()
# number of observations
nobservations <- 2000
#load(file = paste0(prefix,"toggleswitchdata.RData"))
#obs <- obs[1:nobservations]
obs <- t(matrix(read.table("x_cauchy_0_loc.txt")$V1))
obs_sorted <- sort(obs)

# function to compute distance between observed data and data generated given theta
compute_d = function(y){
  sort_y = sort(y)
  mean(abs(sort_y-obs_sorted))
}

target$simulate <- function(theta) matrix(target$robservation(nobservations, theta, target$parameters, target$generate_randomness(nobservations)), nrow = 1)

y_sim <- target$simulate(target$rprior(1, target$parameters))
compute_d(y_sim)

param_algo <- list(nthetas = 300, nmoves = 1, proposal = mixture_rmixmod(),
                   minimum_diversity = 0.5, R = 2, maxtrials = 1000)

filename <- paste0(prefix, "toggleswitchwsmc.n", nobservations, ".RData")
results <- wsmc(compute_d, target, param_algo, savefile = filename, maxsimulation = 1e06) 
load(filename)

####
results_was = results
wsmc.df = wsmc_to_dataframe(results_was)
nsteps = tail(wsmc.df$step,n=1)
#step = nsteps
# set step such that the number of model simulations until that point is about 2.4^10^6
step = nsteps
wsmc.df = wsmc.df[wsmc.df$step == step,]
#results <- wsmc_continue(results, savefile = filename, maxsim = 1e6)
# plot_bivariate(results, 1, 2)