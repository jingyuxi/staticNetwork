using Graphs, Random

function degree_preserving_randomization!(g::SimpleGraph, n_swaps::Int=1000)
    degrees_before = [degree(g, v) for v in 1:nv(g)]

    for _ in 1:n_swaps
        edges_list = collect(edges(g))  # refresh edge list every time for safety
        edge1 = rand(edges_list)
        edge2 = rand(edges_list)

        u1, v1 = src(edge1), dst(edge1)
        u2, v2 = src(edge2), dst(edge2)

        # Ensure all vertices are distinct and edges are different
        if length(Set([u1, v1, u2, v2])) == 4
            # Avoid creating self-loops or duplicate edges
            if !has_edge(g, u1, v2) && !has_edge(g, u2, v1) &&
               u1 != v2 && u2 != v1

                # Remove old edges
                rem_edge!(g, u1, v1)
                rem_edge!(g, u2, v2)

                # Add new edges
                add_edge!(g, u1, v2)
                add_edge!(g, u2, v1)

                # Check degree preservation
                degrees_after = [degree(g, v) for v in 1:nv(g)]
                if degrees_before != degrees_after
                    # Undo if degree changed
                    rem_edge!(g, u1, v2)
                    rem_edge!(g, u2, v1)
                    add_edge!(g, u1, v1)
                    add_edge!(g, u2, v2)
                end
            end
        end
    end

    degrees_after = [degree(g, v) for v in 1:nv(g)]
    if degrees_before == degrees_after
        println("Degrees were successfully preserved.")
    else
        println("Warning: Degrees were not preserved!")
    end

    return g
end

# --- Example Usage ---

# Create an Erdős–Rényi graph with 100 nodes and edge probability 0.05
g = erdos_renyi(SimpleGraph, 100, 0.05)

println("Degrees before randomization:")
for v in 1:nv(g)
    println("Vertex $v: degree = ", degree(g, v))
end

degree_preserving_randomization!(g, 1000)

println("\nDegrees after randomization:")
for v in 1:nv(g)
    println("Vertex $v: degree = ", degree(g, v))
end