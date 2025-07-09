# network setup

#using LightGraphs

using GraphMakie      
using Graphs           
using CairoMakie       
using Cairo
using GraphMakie.NetworkLayout

n = 10
g = cycle_graph(n)
g_full = complete_graph(n)


fig1, ax, p = graphplot(g, layout =Shell()) 
fig2, ax, p = graphplot(g_full, layout =Shell()) 

hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()

display(fig1)