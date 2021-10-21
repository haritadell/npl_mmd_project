library(winference)
registerDoParallel(cores = detectCores())
rm(list = ls())
setmytheme()

set.seed(11)

doRun <- FALSE
max_time <- 30*60
d <- 2
target <- get_multivariate_normal(d)
target$parameters$tau <- 5
nobservations <- 200
p <- 1
true_theta <- c(1,1)
prefix <- ""

obsfile <- paste0(prefix, "tttmvnormaldata.d", d, ".n", nobservations, ".RData")
obs <- target$robservation(nobservations, true_theta, target$parameters)
save(true_theta, obs, file = obsfile)
