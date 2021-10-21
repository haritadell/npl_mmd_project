library(winference)
setwd("~/Dropbox/mmd_project_code/")
run_exp_gaus <- function(data_path,n,d,n_particles, n_cont, r){
  registerDoParallel(cores = detectCores())
  rm(list = ls())
  setmytheme()
  
  set.seed(11)
  
  doRun <- FALSE
  max_time <- 30*60
  #d <- 4
  target <- get_multivariate_normal(d)
  target$parameters$tau <- 1
  nobservations <- n
  #nparticles 
  p <- 1
  prefix <- ""
  
  #obsfile <- paste0(prefix, "mvnormaldata.d", d, ".n", nobservations, ".RData")
  #load(obsfile)
  
  obs <- t(matrix(read.table(data_path)$V1))
  #obs <- do.call(rbind, lapply(obs, as.numeric))
  #obs <- read.table("data_1_200.txt")
  
  
  # function to simulate data
  target$simulate <- function(theta){
    return(target$robservation(nobservations, theta, target$parameters))
  }
  # wasserstein distance
  wdistance <- get_transport_to_y(obs, p = p)
  #
  param_algo <- list(nthetas = n_particles, nmoves = 1, proposal = mixture_rmixmod(),
                     minimum_diversity = 0.5, R = 2, maxtrials = 1e5)
  
  filename <- paste0(prefix, "mvnormalwsmc.d", d, ".n", nobservations, "outl.",n_cont, ".wasserstein.RData")
  results <- wsmc(wdistance, target, param_algo, maxsimulation = 10^6, savefile = filename)
  load(filename)
  
  results_wasserstein <- results
  sum(results_wasserstein$ncomputed)
  wsmc.wasserstein.df <- wsmc_to_dataframe(results_wasserstein) %>% filter(step == length(results_wasserstein$thetas_history))
  string <- paste0("/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/Gaussian_location_model/WABC/thetas_wabc_outl_", n_cont, "_run_", r, "_dim_",d,".csv")
  write.csv(wsmc.wasserstein.df, string)
}

run_exp_gnk <- function(data_path,n,nparticles, n_cont, r){
  registerDoParallel(cores = detectCores())
  rm(list = ls())
  setmytheme()
  set.seed(11)
  target <- get_gandk()
  # number of observations
  nobservations <- n
  prefix = ""
  #load(paste0(prefix, "gandkdata.RData"))
  #obs <- obs[1:nobservations]
  obs <- t(matrix(read.table(data_path)$V1))
  sort_obs = sort(obs)
  
  compute_d = function(y){
    sort_y = sort(y)
    mean(abs(sort_y-sort_obs))
  }
  
  target$simulate <- function(theta){
    return(matrix(target$robservation(nobservations, theta, target$parameters, target$generate_randomness(nobservations)), nrow = 1))
  }
  
  param_algo <- list(nthetas = 500, nmoves = 1, proposal = mixture_rmixmod(),
                     minimum_diversity = 0.5, R = 2, maxtrials = 100000)
  #t = proc.time()
  #start.time <- Sys.time()
  filename <- paste0(prefix, "gandkwsmc.n", nobservations,"outl.",n_cont, ".RData")
  results <- wsmc(compute_d, target, param_algo, savefile = filename, maxsim = 2.4e6)
  #end.time <- Sys.time()
  #time.taken <- end.time - start.time
  #print(time.taken)
  load(filename)
  results_was = results
  wsmc.df = wsmc_to_dataframe(results_was)
  nsteps = tail(wsmc.df$step,n=1)
  #step = nsteps
  # set step such that the number of model simulations until that point is about 2.4^10^6
  step = nsteps
  wsmc.df = wsmc.df[wsmc.df$step == step,]
  string <- paste0("/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/G_and_k_model/WABC/thetas_wabc_outl_",n_cont,"_run_", r, "new2.csv")
  write.csv(wsmc.df, string)
}

run_exp_togswitch <- function(data_path,n,nparticles, n_cont, r){
  registerDoParallel(cores = detectCores())
  rm(list = ls())
  setmytheme()
  set.seed(11)
  
  prefix = ""
  
  target <- get_toggleswitch()
  # number of observations
  nobservations <- n
  #load(file = paste0(prefix,"toggleswitchdata.RData"))
  #obs <- obs[1:nobservations]
  obs <- t(matrix(read.table(data_path)$V1))
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
  string <- paste0("/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/Toggle_switch_model/WABC/thetas_wabc_run_", r, ".csv")
  write.csv(wsmc.df, string)
}

run_exp_gaus_mmd <- function(data_path,n,d,n_particles, n_cont, r){
  registerDoParallel(cores = detectCores())
  rm(list = ls())
  setmytheme()
  
  set.seed(11)
  
  doRun <- FALSE
  max_time <- 30*60
  #d <- 4
  target <- get_multivariate_normal(d)
  target$parameters$tau <- 1
  nobservations <- n
  #nparticles 
  p <- 1
  prefix <- ""
  
  #obsfile <- paste0(prefix, "mvnormaldata.d", d, ".n", nobservations, ".RData")
  #load(obsfile)
  
  obs <- t(matrix(read.table(data_path)$V1))
  #obs <- do.call(rbind, lapply(obs, as.numeric))
  #obs <- read.table("data_1_200.txt")
  
  
  # function to simulate data
  target$simulate <- function(theta){
    return(target$robservation(nobservations, theta, target$parameters))
  }
  # wasserstein distance
  #wdistance <- get_transport_to_y(obs, p = p)
  #
  mmd_c <- function(first_term, eps, x, y) {
    .Call('_winference_mmd_c', PACKAGE = 'winference', first_term, eps, x, y)
  }
  
  get_mmd_to_y <- function(y){
    nobs <- ncol(y)
    Cy1 <- cost_matrix_L1(y, y)
    Cy2 <- cost_matrix_L2(y, y)^2
    eps <- median(as.numeric(Cy1))
    k_y <- exp(-Cy2/(2*(eps^2)))
    first_term <- sum(k_y) / (nobs*nobs)
    f <- function(z){
      return(mmd_c(first_term, eps, z, y))
    }
    return(f)
  }
  
  compute_d <- get_mmd_to_y(obs)
  param_algo <- list(nthetas = n_particles, nmoves = 1, proposal = mixture_rmixmod(),
                     minimum_diversity = 0.5, R = 2, maxtrials = 1e5)
  
  filename <- paste0(prefix, "mvnormalwsmc.d", d, ".n", nobservations, "outl.",n_cont, ".wasserstein.RData")
  results <- wsmc(compute_d, target, param_algo, maxsimulation = 10^6, savefile = filename)
  load(filename)
  
  results_wasserstein <- results
  sum(results_wasserstein$ncomputed)
  wsmc.wasserstein.df <- wsmc_to_dataframe(results_wasserstein) %>% filter(step == length(results_wasserstein$thetas_history))
  string <- paste0("/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/Gaussian_location_model/WABC/thetas_mabc_outl_", n_cont, "_run_", r, "_dim_",d,".csv")
  write.csv(wsmc.wasserstein.df, string)
}

