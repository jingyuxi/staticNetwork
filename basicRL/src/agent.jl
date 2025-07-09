# agent setup



#α_vec = [Float64.(0.4)]
α_vec =  [Float64.(0.3)]
β_vec = [Float64.(2)]
σ_vec =  [Float64.(0.1)]
θ_vec = [Float64.(1)]


#fix seed
Random.seed!(666)



#sim setup
repetition = 2
horizon = 200
group_size = 10
Q_initial = 1.25



# #α_vec = [Float64.(0.4)]
# α = Float64.((0.1))
# β  = Float64.(2)
# σ  =  Float64.(0.1)
# θ  = Float64.(1)

