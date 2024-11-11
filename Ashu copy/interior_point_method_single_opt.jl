using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
include("utils.jl")
include("two_cell_functions_ctmc.jl")

N = 3
λ = 50.0

lambda_str = replace(string(λ), "." => "_")
folder_name = @sprintf("Interior_Point_Method_results_N_%d_lambda_%s", N, lambda_str)
mkpath(folder_name) 

model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "tol", 1e-8)
set_silent(model) 
# set_optimizer_attribute(model, "print_level", 0)

S,Skeyer,T,TG,TB,Tc=statematrices(N);
ni,np,ns,nt=varioussizes(N)

# set_optimizer_attribute(model, "print_level", 0)

targetstates_good=[target_state+1 for target_state ∈ TG];
targetstates_bad=[target_state+1 for target_state ∈ TB];
targetstates=[targetstates_good;targetstates_bad]
startstates=[start_state+1 for start_state ∈ Tc];
allstates=[startstates;targetstates_good; targetstates_bad]
all_targetstates = vcat(targetstates_good, targetstates_bad)


@variable(model, 0<=P_[1:np] <= 1) 
@variable(model, τ[1:ni]) 
@objective(model, Min, τ[1])
@expression(model, A, hitting_time_mod_give_A(Q_maker_original_mod(P_, N, λ, model, S, Skeyer), 
               targetstates_good, targetstates_bad, allstates, λ, model))
@expression(model, b, hitting_time_mod_give_b(Q_maker_original_mod(P_, N, λ, model, S, Skeyer), 
               targetstates_good, targetstates_bad, allstates, λ, model))
@constraint(model, A * τ == b)
for i in 1:np
    set_start_value(P_[i], 1.0)  # Initial guess for P_
end

for i in 1:ns
    set_start_value(τ[i], 7.0)   # Initial guess for τ
end  
optimize!(model)
P_opt=value.(P_);
tau_opt=value.(τ);

println(termination_status(model),", ", tau_opt[1])

P_opt_ = P_opt .* (abs.(P_opt) .>= 1e-8)
P_opt_ .= min.(P_opt_, 1.0)
Q_opt = Q_maker_using_M(P_opt_, N, λ, S, Skeyer)

println("Is Q irreducible? ", is_irreducible(Q_opt))

Q_filename = generate_filename(folder_name,"Q_matrix_heatmap")
plot_Q_with_colored_yticks(Q_opt, N, all_targetstates, Q_filename,λ)

initial_state = 1
T = 100.0
times, states = simulate_ctmc(Q_opt, initial_state, T)

ctmc_simulation_filename = generate_filename(folder_name,"single_ctmc_simulation")
plot_ctmc_our_problem(times, states, T, N, ctmc_simulation_filename,λ)

num_simulations = 100 
T=1000.0
initial_state = 1

T = 7000.0
longtime_heatmap_simulation_filename = generate_filename(folder_name,"multiple_ctmc_simulation_heatmap_longtime")
plot_ctmc_our_problem_multi(Q_opt, initial_state, T, N, num_simulations, longtime_heatmap_simulation_filename,λ)


invariant_heatmap_simulation_filename = generate_filename(folder_name,"invariant_ctmc_heatmap")
plot_ctmc_invar_distn_our_problem(Q_opt,  N, invariant_heatmap_simulation_filename, λ)
# eig = eigen(Q_opt')
# eigenvalues = eig.values
# eigenvectors = eig.vectors
# real_parts = real.(eigenvalues)
# imag_parts = imag.(eigenvalues)
# tolerance = 1e-8
# essential_zeros = [isapprox(x, 0.0, atol=tolerance) for x in eigenvalues]
# zero_indices = findall(x -> isapprox(x, 0.0, atol=tolerance), eigenvalues)
# println("The eigenvalues of Q_opt are: ", zero_indices)
# pi_vector = real(eigenvectors[:, zero_indices])
# threshold = 1e-8
# pi_vector = abs.(map(x -> abs(x) < threshold ? 0.0 : x, pi_vector))
# pi_vector /= sum(pi_vector)
# println("Steady-state vector π: ", pi_vector)
# pi_matrix=zeros(Float64, N+1, N+1)
# for state_index in 1:ni
#     curr_state = S[state_index - 1] 
#     ua, ub = curr_state[1], curr_state[2]
#     pi_matrix[ua + 1, ub + 1] += pi_vector[state_index]
# end
# println(pi_matrix)

# heatmap(
#     0:N, 0:N, pi_matrix, 
#     xlabel="Cell a", ylabel="Cell b", 
#     title=@sprintf("Invariant Distribution N=%d, λ=%s",N,string(λ)),
#     color=:matter,
#     size=(1500, 1000),
#     xticks=0:1:N,
#     yticks=0:1:N,
#     left_margin=10Plots.mm, right_margin=10Plots.mm, bottom_margin=10Plots.mm
# )

# # Display and save the plot
# display(current())
# # savefig(filename)
# println(rank(eigenvectors))
# println(size(Q_opt, 1))

# scatter(real_parts, imag_parts, xlabel="Real Part", ylabel="Imaginary Part", title="Argand Diagram", legend=false, grid=true)

