using Graphs

# Create a simple graph
g = SimpleGraph(5)
add_edge!(g, 1, 2)
add_edge!(g, 2, 3)
add_edge!(g, 3, 4)
add_edge!(g, 4, 5)
add_edge!(g, 1, 5)

# Print degree sequence before randomization
println("Degrees before randomization: ", degree(g))

# Degree-preserving randomization
randomize!(g)

# Print degree sequence after randomization
println("Degrees after randomization: ", degree(g))

# The graph will be randomized but the degree sequence remains the same.
