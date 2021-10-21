library(glue)
R <- 5 
n <- 2000
d <- 1
n_particles <- 300
for (r in 1:(4)){
    data_path = glue('/Users/HaritaDellaporta/Dropbox/mmd_project_code/data/Toggle_switch_model/run_{r}')
    run_exp_togswitch(data_path,n, n_particles, r)
}