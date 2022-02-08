library(winference)
setwd("~/Dropbox/mmd_project_code/")

run_exp_gnk <- function(data_path,n,nparticles, n_cont, r){
  # data_path: path to the stored observed data 
  # n: number of data points 
  # nparticles: number of theta samples obtained
  # n_cont: percentage of contamination in the data set (for saving file purposes)
  # r: which independent run is being run (for saving file purposes)
  registerDoParallel(cores = detectCores())
  rm(list = ls())
  setmytheme()
  set.seed(11)
  target <- get_gandk()
  ## number of observations
  nobservations <- n
  prefix = ""
  ## load observation from data path
  obs <- t(matrix(read.table(data_path)$V1))
  sort_obs = sort(obs)
  ## Set distance function to be used for ABC
  compute_d = function(y){
    sort_y = sort(y)
    mean(abs(sort_y-sort_obs))
  }
  
  ## Set targets
  target$simulate <- function(theta){
    return(matrix(target$robservation(nobservations, theta, target$parameters, target$generate_randomness(nobservations)), nrow = 1))
  }
  
  ## Set algorithm parameters
  param_algo <- list(nthetas = nparticles, nmoves = 1, proposal = mixture_rmixmod(),
                     minimum_diversity = 0.5, R = 2, maxtrials = 100000)
  #t = proc.time()
  filename <- paste0(prefix, "gandkwsmc.n", nobservations,"outl.",n_cont, ".RData")
  results <- wsmc(compute_d, target, param_algo, savefile = filename, maxsim = 2.4e6)
  # t = proc.time() - t
  # print(t)
  
  ## Save results to data frame
  load(filename)
  results_was = results
  wsmc.df = wsmc_to_dataframe(results_was)
  nsteps = tail(wsmc.df$step,n=1)
  # set step such that the number of model simulations until that point is about 2.4^10^6
  step = nsteps
  wsmc.df = wsmc.df[wsmc.df$step == step,]
  ## save results
  string <- paste0("/results/G_and_k_model/WABC/thetas_wabc_outl_",n_cont,"_run_", r, ".csv")
  write.csv(wsmc.df, string)
}