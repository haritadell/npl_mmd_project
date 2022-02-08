#+ presets, echo = FALSE, warning = FALSE, message = FALSE
library(winference)
registerDoParallel(cores = detectCores())
rm(list = ls())
setmytheme()
set.seed(11)
target <- get_gandk()
# number of observations
nobservations <- 2**11
prefix = ""
#load(paste0(prefix, "gandkdata.RData"))
#obs <- obs[1:nobservations]
#obs <- t(matrix(read.table("data_0_gnk.txt")$V1))
r <- 1
n_cont <- 0
data_path = paste0('/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/G_and_k_model/run_', r, '_outl_', n_cont)
obs <- t(matrix(read.table(data_path)$V1))
sort_obs = sort(obs)


#compute_d <- get_hilbert_to_y(matrix(obs, nrow = 1))

# compute_d = function(y){
#   sort_y = sort(y)
#   mean(abs(sort_y-sort_obs))
# }
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

target$simulate <- function(theta){
  return(matrix(target$robservation(nobservations, theta, target$parameters, target$generate_randomness(nobservations)), nrow = 1))
}

# M=10000
# ts = rep(0,M)
# for(i in 1:M){
#   y_sim = target$simulate(target$rprior(1, target$parameters))
#   t=proc.time()
#   compute_d_alt(y_sim)
#   t=proc.time() -t
#   ts[i] = t[3]
# }
# nthetas was 20148
param_algo <- list(nthetas = 2048, nmoves = 1, proposal = mixture_rmixmod(),
                   minimum_diversity = 0.5, R = 2, maxtrials = 100000)
t = proc.time()
for (r in 1:1){
start.time <- Sys.time()
filename <- paste0(prefix, "gandkwsmc.n", nobservations,"outl",.0, ".RData")
results <- wsmc(compute_d, target, param_algo, savefile = filename, maxsim = 2.4e6)
end.time <- Sys.time()
time.taken <- end.time - start.time
print(time.taken)
}
load(filename)
#results <- wsmc_continue(results, savefile = filename, maxtime = 14*60*60)


# load(filename)
wsmc.df <- wsmc_to_dataframe(results)
nsteps = tail(wsmc.df$step,n=1)
#step = nsteps
# set step such that the number of model simulations until that point is about 2.4^10^6
step = nsteps
wsmc.df = wsmc.df[wsmc.df$step == step,]
string <- paste0("/Users/HaritaDellaporta/Dropbox/mmd_project_code/results/G_and_k_model/MABC/thetas_wabc_outl_",n_cont,"_run_", r, ".csv")
#write.csv(wsmc.df, string)
# nsteps <- max(wsmc.df$step)
#
# # plot_bivariate_polygon(results, 1, 2)
# # plot_bivariate_polygon(results, 3, 4)
#
# library(gridExtra)
# grid.arrange(plot_marginal_time(results, 1),
#   plot_marginal_time(results, 2),
#   plot_marginal_time(results, 3),
#   plot_marginal_time(results, 4), nrow = 2)
