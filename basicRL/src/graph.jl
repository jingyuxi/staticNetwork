
using Graphs
using GraphMakie
using CairoMakie
using NetworkLayout


# plot g
function plot_g(g::SimpleGraph{Int64})
    layout = Shell()  #call from NetworkLayout

    fig, ax, p = graphplot(g,
        node_size=10,
        node_color=:red,
        edge_width=1,
        layout=layout)

    hidedecorations!(ax)
    hidespines!(ax)
    ax.aspect = DataAspect()
    fig
end

# load static graph
g_t = complete_graph(group_size)
matrix_g_t = Matrix(adjacency_matrix(g_t))

plot_g(g_t)

function adapt_erdos_renyi(group_size::Int, edge_num::Int)

    g = erdos_renyi(group_size, edge_num)

    return Matrix(adjacency_matrix(g))

end


 #matrix_g = adapt_erdos_renyi(group_size, 6)








# #plot 
# layout = Shell()  #call from NetworkLayout

# fig, ax, p = graphplot(g1,
#     node_size=10,
#     node_color=:red,
#     edge_width=1,
#     layout=layout)

# hidedecorations!(ax)
# hidespines!(ax)
# ax.aspect = DataAspect()
# fig












#     matrix_g_t = matrix_g

using Graphs
using Plots
g = barabasi_albert(100, 2)
#plot_g(g)

counts = degree_histogram(g)
# Degree values corresponding to the counts
degrees = 0:length(counts)-1


# else 

# end


#graph generation and conversion to a matrix










#plot simple graph


function plot_g(g::SimpleGraph{Int64})
    layout = Shell()  #call from NetworkLayout

    fig, ax, p = graphplot(g,
        node_size=10,
        node_color=:red,
        edge_width=1,
        layout=layout)

    hidedecorations!(ax)
    hidespines!(ax)
    ax.aspect = DataAspect()
    fig
end


#adaptive random network - basic 
function network_selection(t::Int, adapt_t::Vector{Int})
    if t in adapt_t

        g = barabasi_albert(20, 2)
        plot_g(g)
        println("this is t in adpat")

    else
        g2 = barabasi_albert(20, 10)
        plot_g(g2)
        println("this is t not in adapt")

    end
end

