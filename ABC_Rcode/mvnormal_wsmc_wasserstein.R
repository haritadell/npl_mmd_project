library(winference)
registerDoParallel(cores = detectCores())
rm(list = ls())
setmytheme()

set.seed(11)

doRun <- FALSE
max_time <- 30*60
d <- 4
target <- get_multivariate_normal(d)
target$parameters$tau <- 1
nobservations <- 200
nparticles <- 2^9
p <- 1
prefix <- ""

#obsfile <- paste0(prefix, "mvnormaldata.d", d, ".n", nobservations, ".RData")
#load(obsfile)

#obs <- t(matrix(read.table("data_5_200_gaussian_1.txt")$V1))
r<-0
n_cont<-0
data_path = glue('/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/Gaussian_location_model/run_{r}_outl_{n_cont}_dim_{d}')
obs <- t(matrix(read.table(data_path)$V1))
#obs <- do.call(rbind, lapply(obs, as.numeric))
#obs <- read.table("data_1_200.txt")


# function to simulate data
target$simulate <- function(theta){
  return(target$robservation(nobservations, theta, target$parameters))
}
# wasserstein distance
#wdistance <- get_transport_to_y(obs, p = p)
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
#
param_algo <- list(nthetas = 2048, nmoves = 1, proposal = mixture_rmixmod(),
                   minimum_diversity = 0.5, R = 2, maxtrials = 1e5)
#nparticles
filename <- paste0(prefix, "mvnormalwsmc.d", d, ".n", nobservations, "outl",.5, ".wasserstein.RData")
results <- wsmc(compute_d, target, param_algo, maxsimulation = 10^6, savefile = filename) #wdistance
load(filename)
# results <- wsmc_continue(results, savefile = filename, maxsimulation = 800000)
#
# load(filename)
# plot_threshold_time(results) + geom_point()
# mle <- rowMeans(obs)
# plot_bivariate(results, 1, 2, from = 10) + geom_vline(xintercept = mle[1]) + geom_hline(yintercept = mle[2])
# plot_marginal(results, 1, from = 10)

# library(microbenchmark)
# microbenchmark(wdistance(target$simulate(true_theta)), times = 1000)