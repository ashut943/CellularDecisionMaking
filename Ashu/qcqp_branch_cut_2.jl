using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, DataStructures, MathOptInterface, Printf
const MOI = MathOptInterface
include("utils.jl")
include("two_cell_functions_ctmc.jl")


log_file = open("output_log.txt", "w")

N=5
λ=150.0
upper_bound=Inf
upper_bound_tau_0=Inf
initial_Pval=1.0
initial_tauval=0.0



#Find upper bound
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
    set_start_value(P_[i], initial_Pval)  # Initial guess for P_
end

for i in 1:ns
    set_start_value(τ[i], initial_tauval)   # Initial guess for τ
end  

optimize!(model)
P_opt=value.(P_);
tau_opt=value.(τ);
tau_opt_tilde = tau_opt[startstates]

upper_bound=maximum(tau_opt_tilde)
upper_bound_tau_0=tau_opt[1]

println(termination_status(model),"; ",upper_bound_tau_0,"; ",upper_bound)

S,Skeyer,T,TG,TB,Tc=statematrices(N);
ni,np,ns,nt=varioussizes(N)

Di=D_maker(N,λ,S,Skeyer,T,TG,TB,Tc);
Ei=E_maker(N,λ,S,Skeyer,T,TG,TB,Tc);
S0=Ei[:,:,np+1];
Ci=C_maker(N,λ,S,Skeyer,T,TG,TB,Tc);

U=zeros(nt)
L=zeros(nt)

L[nt]=1.0
for i in 1:nt-1
    if(i<=np)
        U[i]=1.0
    else
        U[i]=upper_bound
    
    end
end
U[nt]=1


# Initialize variables
global global_best_ub_ = upper_bound_tau_0
global global_best_solution_ = nothing
global global_best_lb_ = -Inf
tolerance = 1e-7
tolerance_soln = 1
max_depth = 3
n_intervals = 10

root_bounds = [(L[i], U[i]) for i in 1:nt]
root_node = Node(-Inf, root_bounds, 0)

node_queue = Stack{Node}()
push!(node_queue, root_node)

# Branch and Bound Loop
while !isempty(node_queue)
    global global_best_ub_
    global global_best_lb_
    global global_best_solution_
    current_node = pop!(node_queue)
    
    if current_node.depth > max_depth
        continue
    end

    status, lb, X_relaxed = solve_relaxation(current_node, S0, Ci, Di, Ei, np, ns, nt)
    if lb >= global_best_ub_ - tolerance
        # log_to_file("Best Till now: ($global_best_ub_, $global_best_lb_)")
        continue
    end
    
    if X_relaxed !== nothing
        if lb > global_best_lb_ + tolerance && lb < global_best_ub_
            global_best_lb_ = lb
            global_best_solution_ = X_relaxed
            log_to_file("---->Updated best lower bound: $global_best_lb_")
        end
    end

    if 0.5 * global_best_ub_ - global_best_lb_ > tolerance_soln && current_node.depth < max_depth #can remove the last condition to terminate later perhaps
        var_to_branch = select_branching_variable(current_node.bounds)
        lb_var = current_node.bounds[var_to_branch][1]
        ub_var = current_node.bounds[var_to_branch][2]
        interval_size = (ub_var - lb_var) / n_intervals
        log_to_file("========================================")
        log_to_file("Node Depth: $(current_node.depth); Current Solution: ($global_best_ub_, $lb); $(global_best_ub_ - lb)")
        log_to_file("Current Bounds: ($lb_var, $ub_var, $interval_size)")
        log_to_file("Best Till now: ($global_best_ub_, $global_best_lb_)")
        log_to_file("========================================")

        for i in 0:(n_intervals - 1)
            log_to_file("Current child number: $i")
            child_lb = lb_var + i * interval_size
            child_ub = lb_var + (i + 1) * interval_size

            if i == n_intervals - 1
                child_ub = ub_var
            end

            child_bounds = copy(current_node.bounds)
            child_bounds[var_to_branch] = (child_lb, child_ub)
            child_node = Node(-Inf, child_bounds, current_node.depth + 1)
            status_child, lb_child, X_relaxed_child = solve_relaxation(child_node, S0, Ci, Di, Ei, np, ns, nt)
            log_to_file("This child: $lb_child")
            if lb_child < global_best_ub_ - tolerance
                push!(node_queue, child_node)
                log_to_file("This child node has been added to the queue")
                if X_relaxed_child !== nothing && lb_child > global_best_lb_ + tolerance
                    global_best_lb_ = lb_child
                    global_best_solution_ = X_relaxed_child
                    log_to_file("--->Updated best lower bound from child: $global_best_lb_")
                end
            end
        end
    end
end

log_to_file("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
log_to_file("========================================")
log_to_file("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Output the best solution found
if global_best_solution_ !== nothing
    log_to_file("\nBest(*) objective value Upper bound: $global_best_ub_")
    log_to_file("\nBest(*) objective value lower bound: $global_best_lb_")
    log_to_file("Rank of the solution matrix X: $(rank(global_best_solution_))")
else
    log_to_file("No feasible solution found.")
end
close(log_file)