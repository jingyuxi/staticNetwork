
# asocial selection function
## Action-Value methods
function qUpdate(reward::Float64, oldEstimate::Float64, stepSize::Float64)
    # Ensure the stepSize is in [0,1]
    if !(0 <= stepSize && stepSize <= 1)
        error("stepsize must belong to [0,1]")
    end
    # update Q values
    newEstimate = oldEstimate + stepSize * (reward - oldEstimate)
    return newEstimate
end

## Action selection function, small temperature more exploitative
function softmaxTemperature(values::Vector{Float64}, temperature::Real)
    # Ensure the temperature is positive 
    if temperature <= 0
        error("Temperature must be positive")
    end

    # Scale Q-values by temperature (more temperature = more exploration)
    scaled_values = values ./ temperature

    # Numerical stability trick: subtract max before exponentiating
    exp_values = exp.(scaled_values .- maximum(scaled_values))

    # Normalize to get probabilities
    softmax_probs = exp_values ./ sum(exp_values)

    return softmax_probs
end

## Action selection function, small temperature more explorative
function softmaxInverseTemperature(values::Vector{Float64}, temperature::Real)
    # Ensure the temperature is positive
    if temperature <= 0
        error("Temperature must be positive")
    end

    # Scale values by the inverse temperature (Q * β)
    scaled_values = values .* temperature

    # Numerical stability: subtract max before exponentiating
    exp_values = exp.(scaled_values .- maximum(scaled_values))

    # Normalize
    softmax_probs = exp_values ./ sum(exp_values)

    return softmax_probs
end


# Social selection function
## Action selection function- social frenqency, controls the strength of conformity
function conformityProbability(frenquencyAction, θ) # frenquencyAction: vector of frenquency of each action

    if all(==(0), frenquencyAction)
        conformity_probabilities = zeros(length(frenquencyAction))
    else
        # Compute frenqency bias
        bias_values = frenquencyAction .^ θ

        # Compute social probabilities
        conformity_probabilities = bias_values ./ sum(bias_values)
    end
    return conformity_probabilities

end

# social RL_probabilities
function choiceProbability(RL_probabilities::Vector{Float64}, Social_probabilities::Vector{Float64}, copyWeight::Float64) # σ copyWeight
    # Ensure the copyWeight is in [0,1]
    if !(0 <= copyWeight && copyWeight <= 1)
        error("copyWeight must belong to [0,1]")
    end

    # compute global probability, (1-σ)RL_probabilities + σ(Social_probabilities)
    global_probability = (1 .- copyWeight) .* RL_probabilities .+ (copyWeight .* Social_probabilities)

    return global_probability
end

#add guassian noise to parameter in (0,1), input vector

function add_noise_logit_transformation_vec(parameter::Vector{Float64})
    #bound the parameter to [1e-5, 1 - 1e-5]
    parameter_bounded = clamp.(parameter, 1e-5, 1 - 1e-5)

    #logit transformation
    parameter_logit = log.(parameter_bounded ./ (1 .- parameter_bounded))

    #add guassian noise, scale to σ
    σ = 0.05
    parameter_logit_noise = parameter_logit .+ σ .* randn(size(parameter_logit))

    #define inverse logit function, condational for numerical stability
    inverse_logit_stability = x_vec -> map(x -> x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (1 + exp(x)), x_vec)
    #convert back to (0,1)
    parameter_noise = inverse_logit_stability(parameter_logit_noise)
    return parameter_noise
end



#add guassian noise to parameter in (0,1), input scalar

function add_noise_logit_transformation_scalar(parameter::Float64)
    # Bound the scalar
    parameter_bounded = clamp(parameter, 1e-5, 1 - 1e-5)

    # Logit transform
    parameter_logit = log(parameter_bounded / (1 - parameter_bounded))

    # Add Gaussian noise
    σ = 0.05
    parameter_logit_noise = parameter_logit + σ * randn()

    # Inverse logit, numerically stable
    if parameter_logit_noise > 0
        parameter_noise = 1 / (1 + exp(-parameter_logit_noise))
    else
        parameter_noise = exp(parameter_logit_noise) / (1 + exp(parameter_logit_noise))
    end

    return parameter_noise
end



#add guassian noise to parameter with no bound, input vector

function add_noise_no_bound_vec(parameter::Vector{Float64})

    σ = 0.05
    parameter_noise = parameter .+ σ .* randn(size(parameter))
    return parameter_noise
end




#add guassian noise to parameter with no bound, input scalar




function add_noise_no_bound_scalar(parameter::Float64)
    σ = 0.05
    return parameter + σ * randn()
end




#3D array to storage Q values, each layer is the Q value of one arm
function Q_value_array(group_size::Int, horizon::Int, num_arms::Int)
    Q_array = Array{Union{Missing,Float64}}(missing, group_size, horizon, num_arms)

    return Q_array
end

#3D array for action probability 
function action_prob_global_array(group_size::Int, horizon::Int, num_arms::Int)
    prob_global_array = Array{Union{Missing,Float64}}(missing, group_size, horizon, num_arms)
    prob_global_array .= missing
    return prob_global_array
end



#update action based on global choice probability
function action_based_on_Prob(i::Int, t::Int, num_arms::Int, prob_global_array)
    prob_vec = Float64.(replace(collect(prob_global_array[i, t, :]), missing => 0.0))
    action = sample(1:num_arms, Weights(prob_vec))
    return Int(action)
end


#update reward based on actions, sampling from dist_array generate from function arm_distribution()
function reward_based_on_action(action_t::Vector{Int})

    reward_t = rand.(dist_array[action_t])
    return reward_t
end

#update Q_array with Action-Value method qUpdate()
function Q_update!(
    Q_array,
    action_t::Vector{Int},
    reward_t::Vector{Float64},
    t::Int,
    group_size::Int,
    stepSize::Float64
)
    Q_array[:, t+1, :] = Q_array[:, t, :]
    for i in 1:group_size

        Q_array[i, t+1, action_t[i]] = qUpdate(reward_t[i], Q_array[i, t, action_t[i]], stepSize)
    end

    return Q_array

end

#update softmax_array calculate asocial prbability based on function softmaxInverseTemperature()
function softmax_update(
    Q_array,
    temperature::Real,
    t::Int,
    group_size::Int,
    num_arms::Int)
    softmax_array = Array{Union{Missing,Float64}}(undef, group_size, num_arms)
    for i in 1:group_size
        Q_i = vec(Q_array[i, t+1, :])
        if any(ismissing, Q_i)
            error("Vector contains missing values!")
        else
            Q_vec = Float64.(Q_i)
            softmax_probs_updated = softmaxInverseTemperature(Q_vec, temperature)
            softmax_array[i, :] = softmax_probs_updated
        end
    end
    return softmax_array

end

#socialFrequency of this t, static network
# g_t graph of this t, global action_array
function socialFrequency(
    action_array,
    g_t,
    t::Int,
    group_size::Int,
    r::Int)
    # prepare 2D array to storage frequency of this t, col1 = t, col2 = index of nodes, col3 = actions(0 = no edge), col4 = count, col5 = repetition
    freq_array_t = Array{Union{Missing,Int}}(undef, 0, 5)
    for i in 1:group_size
        #perceived action vector of vi
        action_vec = Int[]
        for j in 1:group_size

            if g_t[i, j] == 1

                push!(action_vec, action_array[j, t])
            else
                push!(action_vec, 0) # 0 = no edge
            end

        end

        unique_vals = collect(0:num_arms)
        # 2D array for frequency of vi: step, v index, action index, frequency, by column 
        freq_array = Array{Union{Missing,Int}}(undef, length(unique_vals), 5)
        freq_array[:, 1] .= t # Fill this column with step index
        freq_array[:, 2] .= i # Fill this column with node index
        freq_array[:, 3] .= unique_vals # Fill this column with unique actions, 0 = no edges
        freq_array[:, 5] .= r # Fill this column with repetition index

        freq_vec = Int[]
        for val in unique_vals

            push!(freq_vec, count(==(val), action_vec))
            #print(freq_vec)

        end
        freq_array[:, 4] .= freq_vec  # Fill this column with frequency counts

        # frequency array of all nodes at time t, col1 = t, col2 = index of nodes, col3 = actions(0 = no edge), col4 = count, col5 = repetition
        freq_array_t = vcat(freq_array_t, freq_array)
        # freq_array_t = hcat(freq_array_t, repeat([t], length(freq_array_t)))

    end
    return freq_array_t
end


#socialFrequency of this t, static network
# g_t graph of this t, global action_array
function socialFrequency_ineffcient(
    action_array::Matrix{Union{Missing,Int}},
    g_t::Matrix{Int},
    t::Int,
    group_size::Int,
    r::Int,
    num_arms::Int)
    # prepare 2D array to storage frequency of this t, col1 = t, col2 = index of nodes, col3 = actions(0 = no edge), col4 = count, col5 = repetition
    freq_array_t = Array{Union{Missing,Int}}(undef, 0, 5)
    for i in 1:group_size
        #perceived action vector of vi
        action_vec = Int[]
        for j in 1:group_size

            if g_t[i, j] == 1

                push!(action_vec, action_array[j, t])
            else
                push!(action_vec, 0) # 0 = no edge
            end

        end

        unique_vals = collect(0:num_arms)
        # 2D array for frequency of vi: step, v index, action index, frequency, by column 
        freq_array = Array{Union{Missing,Int}}(undef, length(unique_vals), 5)
        freq_array[:, 1] .= t # Fill this column with step index
        freq_array[:, 2] .= i # Fill this column with node index
        freq_array[:, 3] .= unique_vals # Fill this column with unique actions, 0 = no edges
        freq_array[:, 5] .= r # Fill this column with repetition index

        freq_vec = Int[]
        for val in unique_vals

            push!(freq_vec, count(==(val), action_vec))
            #print(freq_vec)

        end
        freq_array[:, 4] .= freq_vec  # Fill this column with frequency counts

        # frequency array of all nodes at time t, col1 = t, col2 = index of nodes, col3 = actions(0 = no edge), col4 = count, col5 = repetition
        freq_array_t = vcat(freq_array_t, freq_array)
        # freq_array_t = hcat(freq_array_t, repeat([t], length(freq_array_t)))

    end
    return freq_array_t
end





#socialFrequency of this t, static network
# g_t graph of this t, global action_array
# proallocation for optimized performance

function socialFrequency(
    action_array::Matrix{Union{Missing,Int}},
    g_t::Matrix{Int},
    t::Int,
    group_size::Int,
    r::Int,
    num_arms::Int
)
    # Preallocate for maximum possible rows: group_size × (num_arms + 1)
    total_rows = group_size * (num_arms + 1)
    freq_array_t = Array{Union{Missing,Int}}(undef, total_rows, 5)

    row_start = 1

    for i in 1:group_size
        # perceived actions of neighbors
        action_vec = [g_t[i, j] == 1 ? action_array[j, t] : 0 for j in 1:group_size]

        # count 0 to num_arms (0 means no edge)
        unique_vals = collect(0:num_arms)
        freq_vec = [count(==(val), action_vec) for val in unique_vals]
        len_vals = length(unique_vals)

        # Fill rows in-place
        freq_array_t[row_start:row_start+len_vals-1, 1] .= t
        freq_array_t[row_start:row_start+len_vals-1, 2] .= i
        freq_array_t[row_start:row_start+len_vals-1, 3] .= unique_vals
        freq_array_t[row_start:row_start+len_vals-1, 4] .= freq_vec
        freq_array_t[row_start:row_start+len_vals-1, 5] .= r

        row_start += len_vals
    end

    # Trim unused rows (in case of underutilization)
    return freq_array_t[1:row_start-1, :]
end



#p_social calculated based on freq_array_t: socialFrequency of this t, static network
function socialProbability(freq_array_t, σ, θ, group_size::Int, num_arms::Int)

    p_social_array = Array{Union{Missing,Float64}}(missing, group_size, num_arms)
    for i in 1:group_size


        #select frenqency of node_i, the action order is sorted:1,2,...n
        freq_action = freq_array_t[(freq_array_t[:, 2].==i).&(freq_array_t[:, 3].!=0), 4]

        if isempty(freq_action)
            p_social_vi = fill(0.0, num_arms)  # or uniform
        else
            p_social_vi = conformityProbability(freq_action, θ) .* σ
        end
        #p_social_vi = (conformityProbability(freq_action, 1)) .* σ

        p_social_array[i, :] = p_social_vi


    end
    return p_social_array
end
