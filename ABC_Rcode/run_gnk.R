library(glue)
R <- 10 
n <- 2^11
n_particles <- 2^9
for (r in 0:(R-1)){
  for (n_cont in 0:2){
    data_path = glue('/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/G_and_k_model/run_{r}_outl_{n_cont}')
    run_exp_gnk(data_path, n, n_particles, n_cont, r)
  }
}