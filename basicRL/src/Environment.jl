#guassian (μ, σ) 
arm_means = Float64[1, 10]
arm_SDs = Float64[0.5, 0.5]
num_arms = length(arm_means)

#generate payoff distribution array of all arms, dist_array[1] is k1(μ, σ) 
function arm_distribution(arm_means::Vector{Float64}, sd::Vector{Float64})
    dist_array = [Normal(μ, σ) for (μ, σ) in zip(arm_means, arm_SDs)]
    return dist_array
end
dist_array = arm_distribution(arm_means, arm_SDs)


# rand(dist_array[1])
# rand(dist_array[2])