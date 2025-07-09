# arm bandit setup
actions = 2 #number of options in one state
#action value is drawn from distributions, define distributions
mean_vec = [1, 1.5]
sd_vec = [0.03, 1]
collect(mean_vec)
collect(sd_vec)




