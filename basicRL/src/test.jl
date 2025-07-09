# load g_t of t = 1
g_t = complete_graph(group_size)
matrix_g_t = Matrix(adjacency_matrix(g_t))


for t in 1:horizon
    print(t)
    if t in adapt_t

        matrix_g_t = adapt_erdos_renyi(group_size, 6)


    end
    show(stdout, "text/plain", matrix_g_t)
    println()
end


matrix_history = Vector{Matrix{Int64}}()


# load g_t of t = 1
g_t = complete_graph(group_size)
matrix_g_t = Matrix(adjacency_matrix(g_t))

for t in 1:horizon
    if t in adapt_t
        matrix_g_t = adapt_erdos_renyi(group_size, 6)
    end
    push!(matrix_history, copy(matrix_g_t))
end



matrix_g_t = Matrix(matrix_history[6])








































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


t = 1




























































































































i = 1
j = 1
t = 1
r = 2




test = repeat([α β σ θ r], group_size, 1)


combined = [hcat(repeat([α β σ θ r], group_size, 1), prob_global_array[:, :, action]) for action in 1:size(prob_global_array, 3)]
combined_array = cat(combined...; dims=3)



prob_global_array_final = prob_global_array_r

1:group_size
A = repeat([1 2 3 4 5], group_size, 1)


A
α = 1


r = 1


















freq_array_t = socialFrequency(
    action_array,
    g_t,
    1,
    group_size,
    r)










collect(sort(0:num_arms))








pwd()






horizon = 1



t = 3


t = 2

action_t = action_based_on_Prob.(1:group_size, Ref(t), Ref(num_arms), Ref(prob_global_array))








#socialFrequency of this t

function socialFrequency(
    action_array,
    g_t,
    t::Int,
    group_size::Int)
    # prepare 2D array to storage frequency of this t, col1 = t, col2 = index of nodes, col3 = actions(0 = no edge), col4 = count
    freq_array_t = Array{Union{Missing,Int}}(undef, 0, 4)
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

        unique_vals = unique(action_vec)
        # 2D array for frequency of vi: step, v index, action index, frequency, by column 
        freq_array = Array{Union{Missing,Int}}(undef, length(unique_vals), 4)
        freq_array[:, 1] .= t # Fill this column with steps
        freq_array[:, 2] .= i # Fill this column with node index
        freq_array[:, 3] .= unique_vals # Fill this column with unique actions, 0 = no edges


        freq_vec = Int[]
        for val in unique_vals

            push!(freq_vec, count(==(val), action_vec))
            #print(freq_vec)

        end
        freq_array[:, 4] .= freq_vec  # Fill this column with frequency counts

        # frequency array of all nodes at time t, col1 = t, col2 = index of nodes, col3 = actions(0 = no edge), col4 = count
        freq_array_t = vcat(freq_array_t, freq_array)
        # freq_array_t = hcat(freq_array_t, repeat([t], length(freq_array_t)))

    end
    return freq_array_t
end



t = 2
Q_array[:, t, :]

Q_array[:, t+1, :]


softmax_probs_updated = softmaxInverseTemperature(Q_vec, β)


softmax_array[i, :] = softmax_probs_updated



test_matrix = [111; 111] * 3




i = 1


#p_social calculated based on freq_array_t: socialFrequency of this t, static network
function socialProbability(freq_array_t, σ, group_size::Int, num_arms::Int)

    p_social_array = Array{Union{Missing,Float64}}(missing, group_size, num_arms)
    for i in 1:group_size


        #select frenqency of node_i, the action order is sorted:1,2,...n
        freq_action = freq_array_t[(freq_array_t[:, 2].==i).&(freq_array_t[:, 3].!=0), 4]

        p_social_vi = (conformityProbability(freq_action, 1)) * σ

        p_social_array[i, :] = p_social_vi


    end
    return p_social_array
end
t = 1



























# prepare container to storage frequency of this t, col1 = t, col2 = index of nodes, col3 = actions(0 = no edge), col4 = count
freq_array_t = Array{Union{Missing,Int}}(undef, 0, 4)
for i in 1:n
    action_vec = Int[]
    for j in 1:n

        if g_t[i, j] == 1

            action_vec = push!(action_vec, action_array[j, t])
        else
            action_vec = push!(action_vec, 0)
        end

    end
    unique_vals = unique(action_vec)

    #2D array for frequency of vi: v index, action index, frequency, by column 
    freq_array = Array{Union{Missing,Int}}(undef, length(unique_vals), 4)
    freq_array[:, 1] = repeat([t], length(unique_vals))
    freq_array[:, 2] = repeat([i], length(unique_vals))
    freq_array[:, 3] = unique_vals

    freq_vec = Int[]
    for val in unique_vals

        freq_vec = push!(freq_vec, count(==(val), action_vec))
        #print(freq_vec)

    end
    freq_array[:, 4] = freq_vec

    freq_array_t = vcat(freq_array_t, freq_array)
    #freq_array_t = hcat(freq_array_t, repeat([t], length(freq_array_t)))

end

print(freq_array_t)









#for i in 1:n
for j in 1:n
    if g_t[i, j] == 1
        action_vec = Int[]
        action_vec = push!(action_array[j, t])
        unique_vals = unique(action_vec)

        #2D array for frequency of vi: v index, action index, frequency, by column 
        freq_array = Array{Union{Missing,Int}}(missing, length(unique_vals), 3)
        freq_array[:, 1] = repeat([i], length(unique_vals))
        freq_array[:, 2] = unique_vals

        freq_vec = Int[]
        for val in unique_vals

            freq_vec = push!(freq_vec, count(==(val), action_vec))
            print(freq_vec)
        end

        freq_array[:, 3] = freq_vec

    end
end



print(freq_array)




test_vec = g_t[i, :]
unique_vals = unique(test_vec)

#2D array for frequency of vi: v index, action index, frequency, by column 
freq_array = Array{Union{Missing,Int}}(missing, length(unique_vals), 3)
freq_array[:, 1] = repeat([i], length(unique_vals))
freq_array[:, 2] = unique_vals

freq_vec = Int[]
for val in unique_vals

    freq_vec = push!(freq_vec, count(==(val), test_vec))
    print(freq_vec)
end

freq_array[:, 3] = freq_vec
































pwd()
cd("/Users/collectiveintelligence/Documents/google drive data backup/presentation/2025 RLDM")
n100k2 = CSV.read("2arm_n100k2.csv", DataFrame)
n100k2.k = fill(02, length(n100k2.bestMeanProb))

testdf = filter(row -> row.t ≥ 30 && row.t ≤ 40 && row.beta == 6, n100k2)






k_vec = [2, 4, 6, 8, 10, 20, 40, 60, 99]

dfs = DataFrame[]
for k in k_vec
    filename = "n100k$(k).csv"
    df = CSV.read(filename, DataFrame)
    df.k = fill(k, nrow(df))
    push!(dfs, df)
end
df = vcat(dfs...)
dfbeta6_last10 = filter(row -> row.t ≥ 30 && row.t ≤ 40 && row.beta == 6 && row.theta != 0.5, df)


dfbeta6_last10_k = filter(row -> row.k == 2, dfbeta6_last10)
data = dfbeta6_last10_k


rep_mean = combine(groupby(data, [:alpha, :beta, :theta, :sigma, :r, :k]), :bestMeanProb => mean => :mean_bestMeanProb_last10)
dist_sd = combine(groupby(rep_mean, [:alpha, :beta, :theta, :sigma, :k]), :mean_bestMeanProb_last10 => std => :value)
dist_mean = combine(groupby(rep_mean, [:alpha, :beta, :theta, :sigma, :k]), :mean_bestMeanProb_last10 => mean => :value)




heatmap_data = dist_mean
CSV.write("heatmap_data.csv", heatmap_data)

using CairoMakie
using DataFrames

# Get unique values
unique_alpha = sort(unique(heatmap_data.alpha))
unique_beta = reverse(sort(unique(heatmap_data.beta)))
unique_theta = sort(unique(heatmap_data.theta))
unique_sigma = reverse(sort(unique(heatmap_data.sigma)))

# Find global maximum std value
global_max = maximum(heatmap_data.value)
rounded_max = ceil(global_max * 10) / 10

nrows = length(unique_beta)   # subplot grid rows
ncols = length(unique_alpha)  # subplot grid columns

fig = Figure(size=(300 * ncols, 300 * nrows))

for (i, β) in enumerate(unique_beta)
    for (j, α) in enumerate(unique_alpha)
        # Filter data for this alpha-beta pair
        subdf = filter(row -> row.alpha == α && row.beta == β, heatmap_data)

        # Pivot: sigma as rows, theta as columns
        #pivot = unstack(subdf, :sigma, :theta, :value)
        pivot = unstack(subdf, :sigma, :theta, :value; combine=mean)
        # Matrix of std values
        Z = Matrix(pivot[:, Not(:sigma)])

        xlabels = unique_theta  # theta (X)
        ylabels = unique_sigma  # sigma (Y)

        ax = Axis(fig[i, j], title="α = $α, β = $β", xlabel="θ", ylabel="σ")
        # Set colorrange to (0, global_max) for all heatmaps
        CairoMakie.heatmap!(ax, xlabels, ylabels, Z; colormap=:viridis, colorrange=(0, 1))
    end
end

# Add colorbar with rounded range and ticks every 0.1
Colorbar(fig[:, end+1],
    label="",
    limits=(0, 1),
    ticks=0:0.2:1)

fig





println(g1)




heatmap_data = dist_sd


using CairoMakie
using DataFrames

# Get unique values
unique_alpha = sort(unique(heatmap_data.alpha))
unique_beta = reverse(sort(unique(heatmap_data.beta)))
unique_theta = sort(unique(heatmap_data.theta))
unique_sigma = sort(unique(heatmap_data.sigma))

# Find global maximum std value
global_max = maximum(heatmap_data.value)
rounded_max = ceil(global_max * 10) / 10

nrows = length(unique_beta)   # subplot grid rows
ncols = length(unique_alpha)  # subplot grid columns

fig = Figure(size=(300 * ncols, 300 * nrows))

for (i, β) in enumerate(unique_beta)
    for (j, α) in enumerate(unique_alpha)
        # Filter data for this alpha-beta pair
        subdf = filter(row -> row.alpha == α && row.beta == β, heatmap_data)

        # Pivot: sigma as rows, theta as columns
        pivot = unstack(subdf, :sigma, :theta, :value)

        # Matrix of std values
        Z = Matrix(pivot[:, Not(:sigma)])

        xlabels = unique_theta  # theta (X)
        ylabels = unique_sigma  # sigma (Y)

        ax = Axis(fig[i, j], title="α = $α, β = $β", xlabel="θ", ylabel="σ")
        # Set colorrange to (0, global_max) for all heatmaps
        CairoMakie.heatmap!(ax, xlabels, ylabels, Z; colormap=:viridis, colorrange=(0, 1))
    end
end

# Add colorbar with rounded range and ticks every 0.1
Colorbar(fig[:, end+1],
    label="",
    limits=(0, 1),
    ticks=0:0.1:1)

fig







using CairoMakie
using DataFrames
using CSV

# Load the data
data = heatmap_data

# Filter for specific alpha and beta (since your data only has alpha=0.6, beta=6)
filtered_data = filter(row -> row.alpha == 0.6 && row.beta == 6, data)

# Prepare the matrix for heatmap
theta_vals = sort(unique(filtered_data.theta))
sigma_vals = sort(unique(filtered_data.sigma), rev=true)  # Reverse to put high sigma at top

# Create matrix where rows are sigma, columns are theta
value_matrix = Matrix(unstack(filtered_data, :sigma, :theta, :value)[:, Not(:sigma)])

# Create the heatmap
fig = Figure(resolution=(800, 600))
ax = Axis(fig[1, 1],
    title="Heatmap (α=0.6, β=6)",
    xlabel="θ",
    ylabel="σ")

# Plot the heatmap
hm = heatmap!(ax, theta_vals, sigma_vals, value_matrix,
    colormap=:viridis)

# Add colorbar
Colorbar(fig[1, 2], hm, label="")

# Customize ticks
ax.xticks = 0:1:8  # Show every integer theta value
ax.yticks = 0:0.2:1  # Show sigma values in 0.2 increments

fig





































cd("/Users/collectiveintelligence/Documents/Julia")


pwd()

cd("./staticNetwork")























cd("/Users/collectiveintelligence/Documents/sim25/R/2 arm/noRisk/2 arm/n50/k32/n50k32")
n50norisk = CSV.read("n50k32norisk.csv", DataFrame)

n50norisk_last10 = filter(:t => x -> 91 <= x <= 100, n50norisk)


data = n50norisk_last10


rep_mean = combine(groupby(data, [:alpha, :beta, :theta, :sigma, :r]), :bestMeanProb => mean => :mean_bestMeanProb_last10)

dist01 = filter(row -> row.alpha == 0.5 && row.beta == 3 && row.theta == 4 && row.sigma == 0.6, rep_mean)
dist02 = filter(row -> row.alpha == 0.5 && row.beta == 3 && row.theta == 8 && row.sigma == 0.8, rep_mean)

histogram(dist01.mean_bestMeanProb_last10, bins=30, normalize=true, label="Histogram", title="Distribution of Vector")
histogram(dist02.mean_bestMeanProb_last10, bins=30, normalize=true, label="Histogram", title="Distribution of Vector")



using StatsPlots
using Random




density!(dist02.mean_bestMeanProb_last10, label="Density")



# Histogram + density overlay
histogram(dist01.mean_bestMeanProb_last10, bins=30, normalize=true, label="Histogram", alpha=0.4)
StatsPlots.density!(dist01.mean_bestMeanProb_last10, label="Density", linewidth=2)

# Histogram + density overlay
histogram(dist02.mean_bestMeanProb_last10, bins=30, normalize=true, label="Histogram", alpha=0.4)
StatsPlots.density!(dist02.mean_bestMeanProb_last10, label="Density", linewidth=2)






# st in ab
fig = Figure(size=(1400, 1000))
Label(fig[0, 0], "n50k$(degree)GMM")
grid = GridLayout()
# 1. First create all axes and store them
ax_array = [Axis(fig[i, j]) for i in 1:length(β_value), j in 1:length(α_value)]

# 4. Fill each axis
for (i, β) in enumerate(reverse(β_value))
    for (j, α) in enumerate(α_value)
        ax = ax_array[i, j]

        plot_df = filter(row -> row.alpha == α && row.beta == β && row.chunk == 100 && row.k == degree, dataframe_full)

        # Find unique sigma values (we're now plotting different sigma lines)
        sigmas = unique(plot_df.sigma)
        colors = Makie.wong_colors()[1:length(sigmas)]

        # Reverse sigma order for consistent coloring
        sigmas = reverse(sigmas)

        for (k, sig) in enumerate(sigmas)
            mask = plot_df.sigma .== sig

            # Plot theta on X-axis, mean on Y-axis
            CairoMakie.scatter!(
                ax,
                plot_df.theta[mask],
                plot_df.mean[mask];
                color=colors[k],
                markersize=10,
                label="σ = $(sig)"
            )

            # Build Vec4 points: center + symmetric error
            pts = Vec4f.(
                plot_df.theta[mask],
                plot_df.mean[mask],
                plot_df.sd[mask],
                plot_df.sd[mask]
            )

            # Plot error bars
            CairoMakie.errorbars!(
                ax,
                pts;
                color=colors[k],
                linewidth=2
            )
        end

        ax.title = "α = $(α), β = $(β)"
        ax.xlabel = "θ"
        ax.ylabel = "performance"
        CairoMakie.xlims!(ax, minimum(plot_df.theta), maximum(plot_df.theta))
        CairoMakie.ylims!(ax, 0, 1)
    end
end

Legend(fig[2, 6], ax)

fig

save("n50k$(degree)GMM.png", fig)













































































g1 = complete_graph(10)

matrix_g1 = Matrix(adjacency_matrix(g1))

g2 = barabasi_albert(10, 4)

matrix_g2 = Matrix(adjacency_matrix(g2))


typeof(g1)
plot_g(g1)


#adaptive random network -
function network_selection(t::Int, adapt_t::Vector{Int})
    if t in adapt_t

        g = erdos_renyi(100, 0.009, is_directed=false, seed=123)

        plot_g(g)
        println("this is t in adpat")

    else
        g = erdos_renyi(100, 0.009, is_directed=false, seed=123)


        g2 = degree_preserving_randomization!(g, 1000)
        plot_g(g2)
        println("this is t not in adapt")

    end
end



network_selection(t, adapt_t)


t = 2






erdos_renyi(10, 0.5, is_directed=true, seed=123)






























#test t = 1
t = 1
#asocial learning
#take an action based on global probability  
#action_t = [action_function(i, t, num_arms, prob_global_array) for i in 1:group_size]
#broadcating instead of loops, Ref() treat(...) as constants with no broadcasting
#update action based on global choice probability
action_t = action_based_on_Prob.(1:group_size, Ref(t), Ref(num_arms), Ref(prob_global_array))
#record vector action_t of t in action_array
action_array[:, t] .= action_t
#update reward based on actions, sampling from dist_array generate from function arm_distribution()
reward_t = reward_based_on_action(action_t)
#record vector reward_t of t in reward_array
reward_array[:, t] .= reward_t

#
#if t < horizon
#update asocial learning component
#update Q
qUpdate()

Q_array[:, t, :]


#update softmax
softmaxInverseTemperature()


#update social frenquency



#update strangh

#update social Weights

#update global probability

i = 1
t

reward_t[i]







#function(t::Int, )
function Q_update!(
    Q_array,
    action_t::Vector{Int},
    reward_t::Vector{Float64},
    t::Int,
    group_size::Int,
    stepSize::Float64
)

    for i in 1:group_size

        Q_array[i, t, action_t[i]] = qUpdate(reward_t[i], Q_array[i, t, action_t[i]], stepSize)
    end

    return Q_array

end

Q_update!(Q_array,
    action_t,
    reward_t,
    t,
    group_size,
    0.01)


α = 0.1
β = 3.0
θ = 5.0
σ = 0.5




function softmax_update(
    Q_array,
    temperature::Float64,
    t::Int,
    group_size::Int,
    num_arms::Int)
    softmax_array = Array{Union{Missing,Float64}}(missing, group_size, num_arms)
    for i in 1:group_size
        Q_i = Q_array[i, t, :]
        if any(ismissing, Q_i)
            error("Vector contains missing values!")
        else
            Q_vec = Float64.(vec(Q_i))
            softmax_probs_updated = softmaxInverseTemperature(Q_vec, temperature)
            softmax_array[i, :] = softmax_probs_updated
        end
    end
    return softmax_array
end




softmax_array = softmax_update(
    Q_array,
    β,
    t,
    group_size,
    num_arms)
















#
i = 1
t = 1


action_t = action_based_on_Prob.(1:group_size, Ref(t), Ref(num_arms), Ref(prob_global_array))



using Pkg
Pkg.add("DataFrames")
using DataFrames
using Random
using Distributions
using CSV

# Set seed for reproducibility
Random.seed!(42)

# Parameters
n_subjects = 100_000
trials_per_subject = 20
n_rows = n_subjects * trials_per_subject

# Generate data
subjects = repeat(1:n_subjects, inner=trials_per_subject)
trials = repeat(1:trials_per_subject, outer=n_subjects)
conditions = rand(["A", "B"], n_rows)
response_time = round.(rand(Normal(500, 50), n_rows), digits=1)
accuracy = rand(Binomial(1, 0.8), n_rows)

# Create DataFrame
df = DataFrame(
    subject=subjects,
    trial=trials,
    condition=conditions,
    response_time=response_time,
    accuracy=accuracy
)

# Save to CSV
CSV.write("test.csv", df)


@time CSV.read("test01.csv", DataFrame);

test01 = CSV.read("test01.csv", DataFrame)


@time test02 = CSV.read("short.tsv", DataFrame; delim='\t');





using Random
using Debugger
using Revise

function test_function(x, y)
    test_container = x + y
    #@bp 
    return test_container + 1
end
#Debugger.@enter test_function(1,2)
test01 = test_function(1, 2)
println(test01)
eltype(test01)




# Two functions to compare
function sum_squares_loop(n)
    total = 0
    for i in 0:n-1
        total += i^2
    end
    return total
end

function sum_squares_comp(n)
    return sum([i^2 for i in 0:n-1])
end

# Using @time or @btime to compare performance
println("Timing sum_squares_loop:")
@time sum_squares_loop(10_000)

println("\nTiming sum_squares_comp:")
@time sum_squares_comp(10_000)



using LinearAlgebra

A = [1 2; 3 4; 5 6]

F = svd(A)



























#3D array to storage Q values, each layer is the Q value of one arm
function action_prob_global_array(group_size::Int, horizon::Int, num_arms::Int)
    prob_global_array = Array{Union{Missing,Float64}}(undef, group_size, horizon, num_arms)
    prob_global_array .= missing
    return prob_global_array
end



eltype(arr1)













action_prob_global_array(group_size, horizon, num_arms)




prob_global_array = action_prob_global_array(group_size, horizon, num_arms)

using Pkg
Pkg.add("GraphMakie")





using Graphs
using GraphMakie
using CairoMakie
using NetworkLayout

# Create a complete graph with 6 nodes
g = complete_graph(20)

# Force all nodes into a single ring/shell
layout = Shell()  # or Shell([[1,2,3,4,5,6]])

fig, ax, p = graphplot(g, layout=layout)
hidedecorations!(ax)
hidespines!(ax)
ax.aspect = DataAspect()
fig




using Pkg
Pkg.installed(Pycall.jl)





function degree_preserving_randomization!(g::SimpleGraph, n_swaps::Int=1000)
    # Collect edges of the graph into a list for easier randomization
    local_edges = collect(edges(g))

    # Check the degree of each vertex before randomization
    degrees_before = [degree(g, v) for v in 1:nv(g)]

    for _ in 1:n_swaps
        # Randomly choose two edges (u1, v1) and (u2, v2)
        edge1 = local_edges[rand(1:end)]
        edge2 = local_edges[rand(1:end)]

        u1, v1 = edge1.src, edge1.dst
        u2, v2 = edge2.src, edge2.dst

        # Ensure that the selected edges are not the same and do not create self-loops
        if u1 != u2 && v1 != v2 && u1 != v2 && v1 != u2
            # Check that the new edges do not already exist and do not create self-loops
            if !has_edge(g, u1, v2) && !has_edge(g, u2, v1)
                # Perform the edge swap
                rem_edge!(g, u1, v1)
                rem_edge!(g, u2, v2)

                add_edge!(g, u1, v2)
                add_edge!(g, u2, v1)
            end
        end
    end

    # Check the degree of each vertex after randomization
    degrees_after = [degree(g, v) for v in 1:nv(g)]

    # Verify degree preservation
    if degrees_before != degrees_after
        println("Warning: Degrees were not preserved!")
    else
        println("Degrees were successfully preserved.")
    end

    return g
end

# Example Usage:

# Create a cycle graph with 10 vertices
g = cycle_graph(10)

# Print the degree of each node before randomization
println("Degrees before randomization:")
for v in 1:nv(g)
    println("Vertex $v: degree = ", degree(g, v))
end

# Apply degree-preserving randomization
degree_preserving_randomization!(g, 1000)

# Print the degree of each node after randomization
println("\nDegrees after randomization:")
for v in 1:nv(g)
    println("Vertex $v: degree = ", degree(g, v))
end

import Pkg;
Pkg.add("NamedArrays");

using NamedArrays

# Define a 2x2 matrix
data = [1 2; 3 4]

# Create a NamedArray with row and column names
named_arr = NamedArray(data, (["row1", "row2"], [:A, :B]))

# Print the NamedArray
println(named_arr)



using NamedArrays

# Define parameter lists
beta_list = [3, 4]
alpha_list = [1, 2]
sigma_list = [5, 6]
theta_list = [7, 8]
repetition = 2
horizon = 3

# Compute the number of rows
num_rows = length(beta_list) * length(alpha_list) * length(sigma_list) * length(theta_list) * repetition * horizon
num_cols = 8  # Number of variables


# Create the data structure
data_matrix = Array{Union{Missing,Float64}}(missing, num_rows, num_cols)

# Assign values to each column
data_matrix[:, 1] = repeat(alpha_list, inner=length(beta_list) * length(sigma_list) * length(theta_list) * repetition * horizon)
data_matrix[:, 2] = repeat(beta_list, outer=length(alpha_list), inner=length(sigma_list) * length(theta_list) * repetition * horizon)
data_matrix[:, 3] = repeat(sigma_list, outer=length(beta_list) * length(alpha_list), inner=length(theta_list) * repetition * horizon)
data_matrix[:, 4] = repeat(theta_list, outer=length(beta_list) * length(alpha_list) * length(sigma_list), inner=repetition * horizon)
data_matrix[:, 5] = repeat(collect(1:repetition), outer=length(beta_list) * length(alpha_list) * length(sigma_list) * length(theta_list), inner=horizon)
data_matrix[:, 6] = repeat(collect(1:horizon), outer=length(beta_list) * length(alpha_list) * length(sigma_list) * length(theta_list) * repetition)
data_matrix[:, 7] = fill(missing, num_rows)  # bestMeanProb
data_matrix[:, 8] = fill(missing, num_rows)  # averageReward
#display(data_matrix)

show(stdout, "text/plain", data_matrix)
# Define column names


col_names = [:beta, :alpha, :sigma, :theta, :r, :t, :bestMeanProb, :averageReward]

# Create a NamedArray
named_social_learning_data = NamedArray(data_matrix, (["row$i" for i in 1:num_rows], col_names))

# Print the NamedArray
println(named_social_learning_data)




A = collect(repeat(1:2, inner=2, outer=2))
println(A)




#arm setup
num_arms = 2
arm_means = [1, 1.5]
arm_SDs = [0.5, 0.5]
beta_list = [3, 4]
alpha_list = [1, 2]
sigma_list = [5, 6]
theta_list = [7, 8]
repetition = 2
horizon = 3

function Q_value_array(group_size::Int, horizon::Int, num_arms::Int)
    return fill(missing, group_size, horizon, num_arms)
end


test = Q_value_array(2, 3, 4)
test[:, :, 1]






