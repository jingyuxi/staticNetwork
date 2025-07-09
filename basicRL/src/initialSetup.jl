#fix seed
Random.seed!(666)

#sim setup
repetition = 3
horizon = 200
group_size = 10
Q_initial = 1.25

#the action/arm being targeted
target_action = 2
#RL model setup
α_vec = Float64.(collect(0.1:0.1:0.2))
β_vec = Float64.(collect(1:2:4))
σ_vec = Float64.(collect(0:0.8:0.8))
θ_vec = Float64.(collect(1:1:2))

#containers 
#use named array just for readability, compute the original array.
col_var = ("α", "β", "σ", "θ", "r", "t", "choiceProbability", "averageReward")
num_rows = length(α_vec) * length(β_vec) * length(σ_vec) * length(θ_vec) * repetition * horizon
num_cols = length(col_var)

data_matrix = Array{Union{Missing,Float64}}(missing, num_rows, num_cols)

data_matrix[:, 1] = repeat(α_vec, inner=length(β_vec) * length(σ_vec) * length(θ_vec) * repetition * horizon)
data_matrix[:, 2] = repeat(β_vec, outer=length(α_vec), inner=length(σ_vec) * length(θ_vec) * repetition * horizon)
data_matrix[:, 3] = repeat(σ_vec, outer=length(α_vec) * length(β_vec), inner=length(θ_vec) * repetition * horizon)
data_matrix[:, 4] = repeat(θ_vec, outer=length(α_vec) * length(β_vec) * length(σ_vec), inner=repetition * horizon)
data_matrix[:, 5] = repeat(collect(1:repetition), outer=length(α_vec) * length(β_vec) * length(σ_vec) * length(θ_vec), inner=horizon)
data_matrix[:, 6] = repeat(collect(1:horizon), outer=length(α_vec) * length(β_vec) * length(σ_vec) * length(θ_vec) * repetition)
data_matrix[:, 7] = fill(missing, num_rows)
data_matrix[:, 8] = fill(missing, num_rows)

#force to display the full array, otherwise println()
#show(stdout, "text/plain", data_matrix)
#println()
#println(data_matrix)








