# semi-adapt graph

using Graphs
using GraphMakie
using CairoMakie
using NetworkLayout
function adapt_erdos_renyi(group_size::Int, edge_num::Int)

    g = erdos_renyi(group_size, edge_num)

    return Matrix(adjacency_matrix(g))

end


function semi_adapt_graph(group_size::Int, horizon::Int, adapt_t, edge_num::Int)

    g = complete_graph(group_size)
    matrix_g = Matrix(adjacency_matrix(g))
    matrix_all = Matrix{Int64}[]
 for t in 1:horizon
        if t in adapt_t
            matrix_g = adapt_erdos_renyi(group_size, edge_num)
        end
        push!(matrix_all, copy(matrix_g))
    end
return matrix_all
end


adapt_interval = 1
adapt_t = collect(1:adapt_interval:horizon)
adapt_t = filter(x -> x != 1, adapt_t)



matrix_all = semi_adapt_graph(group_size, horizon, adapt_t, 6)



#matrix_g_t = Matrix(matrix_all[t])

#show(stdout, "text/plain", data_matrix)
#println()









# # load graph of t = 1
# g = complete_graph(group_size)
# matrix_g = Matrix(adjacency_matrix(g))

# # container for all graphs
# matrix_all = Vector{Matrix{Int64}}()


# for t in 1:horizon
#     if t in adapt_t
#         matrix_g = adapt_erdos_renyi(group_size, 6)
#     end
#     push!(matrix_all, copy(matrix_g))
# end

# Matrix(matrix_all[6])
# Matrix(matrix_all[7])
# Matrix(matrix_all[8])


# matrix_g_t = Matrix(matrix_all[t])



# g = complete_graph(group_size)
# matrix_g = Matrix(adjacency_matrix(g))

# matrix_all = Vector{Matrix{Int64}}()

# for t in 1:horizon
#     if t in adapt_t
#         matrix_g = adapt_erdos_renyi(group_size, 6)
#     end
#     push!(matrix_all, copy(matrix_g))
# end