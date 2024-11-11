using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO
include("utils.jl")
include("two_cell_functions_ctmc.jl")

N = 3
λ = 0.1

lambda_str = replace(string(λ), "." => "_")
folder_name = @sprintf("fixed_random_parameter_results_N_%d_lambda_%s", N, lambda_str)
mkpath(folder_name) 

ni, np, ns, nt = varioussizes(N)
S, Skeyer, T, TG, TB, Tc = statematrices(N)

f⁺ = rand(N, 2)
f⁻ = rand(N, 2)
f⁺_flattened = collect(Iterators.flatten(f⁺'))
f⁻_flattened = collect(Iterators.flatten(f⁻'))
g = rand(N+1)
koff = rand(Uniform(0, 1))
P = collect(vcat(f⁺_flattened, f⁻_flattened, g, [koff]))
Q = Q_maker_using_M(P, N, λ, S, Skeyer)

println("Is Q irreducible? ", is_irreducible(Q))

targetstates_good = [target_state + 1 for target_state ∈ TG]
targetstates_bad = [target_state + 1 for target_state ∈ TB]
all_targetstates = vcat(targetstates_good, targetstates_bad)
startstates = [start_state + 1 for start_state ∈ Tc]
allstates = [startstates; targetstates_good; targetstates_bad]

Q_filename = generate_filename(folder_name,"Q_matrix_heatmap")
plot_Q_with_colored_yticks(Q, N, all_targetstates, Q_filename,λ)

initial_state = 1
T = 100.0
times, states = simulate_ctmc(Q, initial_state, T)

ctmc_simulation_filename = generate_filename(folder_name,"single_ctmc_simulation")
plot_ctmc_our_problem(times, states, T, N, ctmc_simulation_filename,λ)

num_simulations = 100 
T=1000.0
initial_state = 1

T = 7000.0
longtime_heatmap_simulation_filename = generate_filename(folder_name,"multiple_ctmc_simulation_heatmap_longtime")
plot_ctmc_our_problem_multi(Q, initial_state, T, N, num_simulations, longtime_heatmap_simulation_filename,λ)

null_vec = nullspace(Q')  # Solve λ * Q = 0
π = null_vec / sum(null_vec) 
println(π)

#visualise heatmap of π