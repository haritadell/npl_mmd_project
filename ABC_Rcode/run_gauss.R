library(glue)
R <- 10 
n <- 200
d <- 4
n_particles <- 500
for (r in 1:(9)){
  for (n_cont in 0:2){
    data_path = glue('/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/Gaussian_location_model/run_{r}_outl_{n_cont}_dim_{d}')
    run_exp_gaus_mmd(data_path, n, d, n_particles, n_cont, r)
  }
}