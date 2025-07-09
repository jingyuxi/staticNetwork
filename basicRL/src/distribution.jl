using Random, Distributions, Plots, StatsPlots

Random.seed!(666)
## non risky
non_risky_01 = Normal(1.616, 0.55)
non_risky_02 = Normal(2.385, 0.55)
non_risky_03 = Normal(3.1, 0.55)
μ_01 = 1.616
μ_02 = 2.385
μ_03 = 3.1
density(Normal(1.616, 0.55))
density(non_risky_03)

using Distributions
#plot(Normal(3,5), fill=(0, .5,:orange))
p_nonrisky = plot(non_risky_01,fill=(0, .3,:blue), linealpha = 0.5,linecolor = :blue, grid = false, legend = false)
vline!([1.616], linealpha = 1,linecolor = :black, linestyle =  :dash )
plot!(non_risky_02,fill=(0, .3,:orange), linealpha = 0.5,linecolor = :orange, grid = false)
vline!([2.385], linealpha = 1,linecolor = :black, linestyle =  :dash )
plot!(non_risky_03,fill=(0, .3,:green), linealpha = 0.5,linecolor = :green, grid = false)
vline!([3.1], linealpha = 1,linecolor = :black, linestyle =  :dash )

savefig(p, "nonrisky.png") 



## risky


risky = Normal(1.5, 1)
safe = Normal(1, 0.3)

p_risky = plot(risky, xlims = (1.25-2, 1.25+2), fill=(0, .3,:orange), linealpha = 0.5,linecolor = :orange, grid = false, legend = false)
vline!([1.5], linealpha = 1,linecolor = :black, linestyle =  :dash )
plot!(safe,fill=(0, .3,:blue), linealpha = 0.5,linecolor = :blue, grid = false)
vline!([1], linealpha = 1,linecolor = :black, linestyle =  :dash )

savefig(p_risky, "risky.png") 

