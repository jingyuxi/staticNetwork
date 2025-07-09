
include("Pkg.jl")
include("agent.jl") # -- testing
#include("initialSetup.jl")
include("environment.jl")
include("functionsRL.jl")
include("semiAdapGraph.jl")









total_time = @elapsed begin



    # Preallocate final array outside loop
    prob_global_array_final = Vector{Array{Float64,3}}()

    # Outer parameter loops
    for α in α_vec, β in β_vec, σ in σ_vec, θ in θ_vec

        for r in 1:repetition
            println("α = ", α, "  β = ", β, "  σ = ", σ, "  θ = ", θ, "  r = ", r)

            α_r = add_noise_logit_transformation_scalar(α)
            β_r = add_noise_no_bound_scalar(β)
            σ_r = add_noise_logit_transformation_scalar(σ)
            θ_r = add_noise_no_bound_scalar(θ)

            # Preallocate arrays
            Q_array = fill(Q_initial, group_size, horizon + 1, num_arms)
            prob_global_array = fill(1 / num_arms, group_size, horizon, num_arms)
            action_array = Array{Union{Missing,Int}}(undef, group_size, horizon)
            reward_array = Array{Float64}(undef, group_size, horizon)



            for t in 1:horizon


                # Broadcast-safe
                prob_view = @view prob_global_array[:, t, :]
                action_t = action_based_on_Prob.(1:group_size, Ref(t), Ref(num_arms), Ref(prob_global_array))
                reward_t = reward_based_on_action(action_t)

                action_array[:, t] .= action_t
                reward_array[:, t] .= reward_t

                if t < horizon
                    Q_array = Q_update!(Q_array, action_t, reward_t, t, group_size, α_r)
                    p_softmax = softmax_update(Q_array, β_r, t, group_size, num_arms)
                    # load g of this t
                    matrix_g_t = Matrix(matrix_all[t])
                    print(t)
                    show(stdout, "text/plain", matrix_g_t)
                    println()
                    freq_array_t = socialFrequency(action_array, matrix_g_t, t, group_size, r, num_arms)
                    p_social = socialProbability(freq_array_t, σ_r, θ_r, group_size, num_arms)

                    # Avoid creating intermediate arrays
                    prob_matrix_t = @. (1 - σ_r) * p_softmax + σ_r * p_social
                    prob_global_array[:, t+1, :] .= prob_matrix_t
                end
            end

            # Create array with metadata + probability
            parameter_block = repeat([α β σ θ r], group_size, 1)
            subject_index = collect(1:group_size)

            prob_global_array_r = cat([
                    hcat(parameter_block, subject_index, prob_global_array[:, :, action])
                    for action in 1:num_arms
                ]...; dims=3)

            push!(prob_global_array_final, prob_global_array_r)
        end
    end
end
# Merge all stored blocks into a single array,  var_col = α β σ θ r v_index t1 t2...tn, dims = action
prob_global_array_final = cat(prob_global_array_final...; dims=1)

println("Time taken: ", total_time, " seconds")







#---------------------------------
#unoptimized
# #empty array to record final results

# total_time = @elapsed begin




#     # load graph matrix
#     g1 = matrix_g
#     # g2 = 
#     # gn =


#     # prob_global_array_final, var_row = α β σ θ r t1 t2...tn, var_col = v1 ... vi, dims = action
#     prob_global_array_final = nothing  # placeholder

#     for α in α_vec
#         for β in β_vec
#             for σ in σ_vec
#                 for θ in θ_vec
#                     for r in 1:repetition
#                         println("α = ", α, "  β = ", β, "  σ = ", σ, "  θ = ", θ, "  r = ", r)
#                         #print("α = ", α, "β = ", β, "σ = ", σ, "θ = ", θ, "r = ", r)
#                         α_this_r = add_noise_logit_transformation_scalar(α)
#                         β_this_r = add_noise_no_bound_scalar(β)
#                         σ_this_r = add_noise_logit_transformation_scalar(σ)
#                         θ_this_r = add_noise_no_bound_scalar(θ)


#                         #3D array for Q update, 1st col is t = 0, initial setup
#                         Q_array = Q_value_array(group_size, horizon + 1, num_arms)
#                         Q_array[:, 1, :] .= Q_initial

#                         #3D array for global action probability, t1 = chance level
#                         prob_global_array = action_prob_global_array(group_size, horizon, num_arms)
#                         prob_global_array[:, 1, :] .= 1 / num_arms

#                         #2D array for action & reward
#                         action_array = Array{Union{Missing,Int}}(undef, group_size, horizon)
#                         reward_array = Array{Union{Missing,Float64}}(undef, group_size, horizon)

#                         #2D array for the probability of the traget action
#                         prob_target_array = prob_global_array[:, :, target_action]

#                         #social frequency 
#                         frequency_global_array = Array{Union{Missing,Int}}(undef, 0, 5)



#                         for t in 1:horizon


#                             #asocial learning
#                             #take an action based on global probability  
#                             #action_t = [action_function(i, t, num_arms, prob_global_array) for i in 1:group_size]
#                             #broadcating instead of loops, Ref() treat(...) as constants with no broadcasting
#                             #update action based on global choice probability
#                             action_t = action_based_on_Prob.(1:group_size, Ref(t), Ref(num_arms), Ref(prob_global_array))
#                             #record vector action_t of t in action_array
#                             action_array[:, t] .= action_t
#                             #update reward based on actions, sampling from dist_array generate from function arm_distribution()
#                             reward_t = reward_based_on_action(action_t)
#                             #record vector reward_t of t in reward_array
#                             reward_array[:, t] .= reward_t

#                             #
#                             if t < horizon
#                                 # update asocial learning component
#                                 # update Q_array
#                                 Q_array = Q_update!(
#                                     Q_array,
#                                     action_t,
#                                     reward_t,
#                                     t,
#                                     group_size,
#                                     α_this_r)

#                                 #update asocial probability
#                                 p_softmax = softmax_update(
#                                     Q_array,
#                                     β_this_r,
#                                     t,
#                                     group_size,
#                                     num_arms)

#                                 # update social learning component
#                                 p_asocial = p_softmax * (1 - σ_this_r)
#                                 # read graph of this t
#                                 g_t = g1
#                                 # update social frenquency of all nodes, freq_array_t, col1 = t, col2 = index of nodes, col3 = actions(0 = no edge), col4 = count, col5 = repetition
#                                 freq_array_t = socialFrequency(
#                                     action_array,
#                                     g_t,
#                                     t,
#                                     group_size,
#                                     r,
#                                     num_arms)

#                                 # record freq_array_t to global container frequency_global_array
#                                 frequency_global_array = vcat(frequency_global_array, freq_array_t)


#                                 p_social = socialProbability(
#                                     freq_array_t,
#                                     σ_this_r,
#                                     θ_this_r,
#                                     group_size,
#                                     num_arms)

#                                 # prob of this t
#                                 prob_matrix_t = p_asocial + p_social # node index by row, action index by col

#                                 #update global probability
#                                 # record prob of this t to prob_global_array
#                                 for n in 1:group_size
#                                     for action in 1:num_arms

#                                         prob_global_array[n, t+1, action] = prob_matrix_t[n, action]

#                                     end
#                                 end

#                             end


#                         end
#                         # add parameter info and rep index, var_col = α β σ θ r v_index t1 t2...tn, dims = action
#                         #repeat([α_this_r β_this_r σ_this_r θ_this_r r], group_size, 1)
#                         parameter_combined = [
#                             hcat(
#                                 repeat([α β σ θ r], group_size, 1),
#                                 collect(1:group_size), prob_global_array[:, :, action]
#                             ) for action in axes(prob_global_array, 3)
#                         ]
#                         prob_global_array_r = cat(parameter_combined...; dims=3)

#                         # record prob_global_array_r to prob_global_array_final, var_col = α β σ θ r v_index t1 t2...tn, dims = action
#                         if prob_global_array_final === nothing
#                             prob_global_array_final = prob_global_array_r
#                         else
#                             prob_global_array_final_combined = [vcat(
#                                 prob_global_array_final[:, :, action],
#                                 prob_global_array_r[:, :, action]
#                             ) for action in axes(prob_global_array_r, 3)]
#                             prob_global_array_final = cat(prob_global_array_final_combined...; dims=3)
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end
# println("Time taken: ", total_time, " seconds")



# prob_global_array_final