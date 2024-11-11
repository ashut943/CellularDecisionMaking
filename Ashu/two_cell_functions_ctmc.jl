using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
include("utils.jl")

function varioussizes(N)
    ni = 4 * (N + 1)^2
    np = 5 * N + 2
    ns  = ni - 12
    nt = np + ns + 1
    return ni,np,ns,nt
end

function statematrices(N)
    ni,np,ns,nt=varioussizes(N)
    S = Dict()
    T = [];
    TG = [];
    TB = [];
    for ua in 0:N
        for ub in 0:N
            for sa in 0:1
                for sb in 0:1
                    index = (2 * sa + sb) * (N + 1)^2 + (N + 1) * ua + ub
                    S[index] = (ua, ub, sa, sb)
                end
            end
        end
    end
    Skeyer = Dict(value => key for (key, value) in S);
    for ua in 0:N
        for ub in 0:N
            for sa in 0:1
                for sb in 0:1
                    i = (2 * sa + sb) * (N + 1)^2 + (N + 1) * ua + ub
                    if ua == N && ub == 0
                        push!(TG, i)
                        push!(T, i)
                    end
                    if ua == 0 && ub == N
                        push!(TG, i)
                        push!(T, i)
                    end
                    if ua == N && ub == N
                        push!(TB, i)
                        push!(T, i)
                    end
                end
            end
        end
    end
    Tc = [i for i in 0:ni-1 if i ∉ T];
    return S,Skeyer,T,TG,TB,Tc
end


function Q_maker(P,N::Int64,λ::Float64,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    Q=zeros(ni,ni)
    for (u, u_) in Iterators.product(values(S), values(S))
        if u == u_
            continue
        end
        i = Skeyer[u]
        j = Skeyer[u_]
        flag = 0
        tempk = 0
        ua, ub, sa, sb = u
        ua_, ub_, sa_, sb_ = u_
    
        if ua_==ua+1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sa + ua + 1
        elseif ua_==ua-1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sa * N + ua + 2 * N 
        elseif ub_==ub+1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sb + ub + 1
        elseif ub_==ub-1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sb * N + ub + 2 * N 
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub + 1
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 2
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua + 1
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 2
        end
        if flag == 1
            Q[i+1, j+1] = P[tempk]
        end
    end

    for i in 1:ni
        # Calculate the sum of each row for Qi
        qi = sum(Q[i, :])
        Q[i,i] = -qi
    end
    return Q
    
end


function hitting_time_mod(Q,targetstates_good,targetstates_bad,startstates,λ,model)
    #ugh for simplicity in writing this up, assume irreducible
    #this is for affexpr Q
    n=size(Q,1)
    A=copy(Q)
    b=-ones(n)
    targetstates=[targetstates_good;targetstates_bad]
    for target_state ∈ targetstates
        A[target_state, :] .= 0.0
        A[target_state, target_state] = 1.0
        if(target_state ∈ targetstates_good)
            b[target_state] = 0.0
        else
            b[target_state] = λ
        end
    end
    for i in 1:n
        if all(Q[i, :] .== 0.0) && i ∉ targetstates
            A[i, :] .= 0.0
            A[i, i] = 1.0
            b[i] = 0.0
        end
    end
    model2 = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model2, "print_level", 0)
    @variable(model2, T[1:n])
    @constraint(model2, A * T == b)
    # Solve the model
    optimize!(model2)

    # Get the hitting times
    T_vals = value.(T)
    # Handle infinite hitting times for unreachable states
    for i in 1:n
        if T_vals[i]==0 && i ∉ targetstates
            T_vals[i] = Inf
        end
    end

    return [T_vals[start_state] for start_state ∈ startstates]
end

function hitting_time_mod_give_A(Q,targetstates_good,targetstates_bad,startstates,λ,model)
    #ugh for simplicity in writing this up, assume irreducible
    n=size(Q,1)
    A=copy(Q)
    b=-ones(n)
    targetstates=[targetstates_good;targetstates_bad]
    for target_state ∈ targetstates
        A[target_state, :] .= 0.0
        A[target_state, target_state] = 1.0
        if(target_state ∈ targetstates_good)
            b[target_state] = 0.0
        else
            b[target_state] = λ
        end
    end
    for i in 1:n
        if all(Q[i, :] .== 0.0) && i ∉ targetstates
            A[i, :] .= 0.0
            A[i, i] = 1.0
            b[i] = 0.0
        end
    end
    return A
end

function hitting_time_mod_give_b(Q,targetstates_good,targetstates_bad,startstates,λ,model)
    #ugh for simplicity in writing this up, assume irreducible
    n=size(Q,1)
    A=copy(Q)
    b=-ones(n)
    targetstates=[targetstates_good;targetstates_bad]
    for target_state ∈ targetstates
        A[target_state, :] .= 0.0
        A[target_state, target_state] = 1.0
        if(target_state ∈ targetstates_good)
            b[target_state] = 0.0
        else
            b[target_state] = λ
        end
    end
    for i in 1:n
        if all(Q[i, :] .== 0.0) && i ∉ targetstates
            A[i, :] .= 0.0
            A[i, i] = 1.0
            b[i] = 0.0
        end
    end
    return b
end


function plot_ctmc_with_index_number(times::Vector{Float64}, states::Vector{Int}, T::Float64, N::Int,filename)
    ni,np,ns,nt=varioussizes(N)

    plot(
        xlabel="Time (t)", 
        ylabel="State (s)", 
        title="CTMC Simulation",
        yticks=collect(1:ni),
        xlim=(0, T), 
        ylim=(0.5, ni+0.5),
        grid=:both,
        size=(1500, 1000),left_margin=10Plots.mm, right_margin=10Plots.mm,bottom_margin=10Plots.mm)
    
    for i in 1:(length(times)-1)
        plot!([times[i], times[i+1]], [states[i], states[i]], linewidth=2, label=false)
        if i < length(times)-1
            plot!([times[i+1], times[i+1]], [states[i], states[i+1]], linestyle=:dash, color=:gray, label=false)
        end
    end
    # display(current())
    savefig(filename * ".png")
    savefig(filename * ".svg")
end



function plot_ctmc_our_problem(times::Vector{Float64}, states::Vector{Int}, T::Float64, N::Int, filename::String,λ::Float64)
    ni, np, ns, nt = varioussizes(N)
    S, Skeyer, T_, TG, TB, Tc = statematrices(N)

    u_a_vals = [S[states[i] - 1][1] for i in 1:length(states)]
    u_b_vals = [S[states[i] - 1][2] for i in 1:length(states)]
    
    all_vals = vcat(u_a_vals, u_b_vals)
    y_min = minimum(all_vals)
    y_max = maximum(all_vals)

    plot(
        xlabel="Time (t)", 
        ylabel="State Values", 
        title=@sprintf("CTMC Simulation - Trajectories of Cells a and b; N=%d, λ=%s",N,string(λ)),
        xlim=(0, T),
        ylim=(y_min, y_max),
        grid=:both,
        size=(1500, 1000),
        left_margin=10Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=10Plots.mm,
        legend=:topright
    )
    
    for i in 1:(length(times) - 1)
        plot!([times[i], times[i + 1]], [u_a_vals[i], u_a_vals[i]], linewidth=2, color=:blue, label=(i == 1 ? "Cell a" : ""))
        if i < length(times) - 1
            plot!([times[i + 1], times[i + 1]], [u_a_vals[i], u_a_vals[i + 1]], linestyle=:dash, color=:blue, label=false)
        end
    end

    for i in 1:(length(times) - 1)
        plot!([times[i], times[i + 1]], [u_b_vals[i], u_b_vals[i]], linewidth=2, color=:red, label=(i == 1 ? "Cell b" : ""))
        if i < length(times) - 1
            plot!([times[i + 1], times[i + 1]], [u_b_vals[i], u_b_vals[i + 1]], linestyle=:dash, color=:red, label=false)
        end
    end

    # Display and save the plot
    # display(current())
    savefig(filename * ".png")
    savefig(filename * ".svg")
end


function plot_ctmc_heatmap_mod(time_points::AbstractVector{Float64}, state_probs::Array{Float64,2}, N::Int, filename::String, λ::Float64)
    ni, np, ns, nt = varioussizes(N)
    S, Skeyer, T_, TG, TB, Tc = statematrices(N)
    
    num_states = size(state_probs, 1)
    println(num_states)

    # Adjusting yticks to match uₐ and u_b states as in plot_ctmc_mod_multi
    yticks_vals = collect(1:ni)
    yticks_labels = [string(S[i-1][1], ", ", S[i-1][2]) for i in 1:ni]

    heatmap(
        time_points,
        yticks_vals,
        state_probs,
        xlabel = "Time",
        ylabel = "(uₐ, u_b)",
        title = @sprintf("CTMC Heatmap Trajectories (Empirical), N=%d, λ=%s",N,string(λ)),
        aspect_ratio = :auto,
        xlims = (time_points[1], time_points[end]),
        ylims = (0.5, num_states + 0.5),
        yticks = (yticks_vals, yticks_labels),
        colorbar = true,
        c = :viridis,
        clim = (0, 1),
        legend = false,
        size = (1500, 1000),
        left_margin = 10Plots.mm, 
        right_margin = 10Plots.mm, 
        bottom_margin = 10Plots.mm
    )
    
    # display(current())
    savefig(filename * ".png")
    savefig(filename * ".svg")
end

function plot_ctmc_our_problem_multi(Q::Array{Float64,2}, initial_state::Int, T::Float64, N::Int, num_simulations::Int, filename::String, λ::Float64)
    ni, np, ns, nt = varioussizes(N)
    S, Skeyer, T_, TG, TB, Tc = statematrices(N)

    visit_counts = zeros(Int, N+1, N+1)

    for _ in 1:num_simulations
        times, states = simulate_ctmc(Q, initial_state, T)
        for state_index in states
            curr_state = S[state_index - 1]
            ua, ub = curr_state[1], curr_state[2]
            visit_counts[ua + 1, ub + 1] += 1
        end
    end
    heatmap(
        0:N, 0:N, log.(visit_counts .+ 1), 
        xlabel="Cell a", ylabel="Cell b", 
        title=@sprintf("Aggregated Frequency of Visits to States across Multiple Simulations N=%d, λ=%s",N,string(λ)),
        color=:matter,
        size=(1500, 1000),
        xticks=0:1:N,   # Show only integer ticks from 0 to N
        yticks=0:1:N,   # Show only integer ticks from 0 to N
        left_margin=10Plots.mm, right_margin=10Plots.mm, bottom_margin=10Plots.mm
    )
    # display(current())
    savefig(filename * ".png")
    savefig(filename * ".svg")
end

function plot_ctmc_invar_distn_our_problem(Q::Array{Float64,2},  N::Int, filename::String, λ::Float64)
    ni, np, ns, nt = varioussizes(N)
    S, Skeyer, T_, TG, TB, Tc = statematrices(N)
    tolerance = 1e-8

    eig = eigen(Q')
    eigenvalues = eig.values
    eigenvectors = eig.vectors
    real_parts = real.(eigenvalues)
    imag_parts = imag.(eigenvalues)

    essential_zeros = [isapprox(x, 0.0, atol=tolerance) for x in eigenvalues]
    zero_indices = findall(x -> isapprox(x, 0.0, atol=tolerance), eigenvalues)

    pi_vector = real(eigenvectors[:, zero_indices])
    threshold = 1e-8
    pi_vector = abs.(map(x -> abs(x) < threshold ? 0.0 : x, pi_vector))
    pi_vector /= sum(pi_vector)
    pi_matrix=zeros(Float64, N+1, N+1)
    for state_index in 1:ni
        curr_state = S[state_index - 1] 
        ua, ub = curr_state[1], curr_state[2]
        pi_matrix[ua + 1, ub + 1] += pi_vector[state_index]
    end

    heatmap(
        0:N, 0:N, pi_matrix, 
        xlabel="Cell a", ylabel="Cell b", 
        title=@sprintf("Invariant Distribution N=%d, λ=%s",N,string(λ)),
        color=:matter,
        size=(1500, 1000),
        xticks=0:1:N,
        yticks=0:1:N,
        left_margin=10Plots.mm, right_margin=10Plots.mm, bottom_margin=10Plots.mm
    )

    # display(current())
    savefig(filename * ".png")
    savefig(filename * ".svg")
    return pi_matrix[N + 1, N + 1] < 0.5
end

function run_pipeline_for_various_lambda(N::Int, lambda_values::Vector{Float64}, num_simulations::Int, initialPval, initialtauval)
    lambda_start = minimum(lambda_values)
    lambda_start_str = replace(string(lambda_start), "." => "_")
    lambda_end = maximum(lambda_values)
    lambda_end_str=lambda_str = replace(string(lambda_end), "." => "_")
    sub_folder = @sprintf("simulation_parameter_results_N_%d_lambdastart_%s_lambdaend_%s", N, lambda_start_str,lambda_end_str)
    overall_folder=joinpath(main_folder,sub_folder)
    if !isdir(overall_folder)
        mkpath(overall_folder)
    end
    movie_folder = @sprintf("movie_folder")
    overall_movie_folder=joinpath(overall_folder,movie_folder)
    if !isdir(overall_movie_folder)
        mkpath(overall_movie_folder)
    end

    movie_folder_2 = @sprintf("movie_folder_2")
    overall_movie_folder_2=joinpath(overall_folder,movie_folder_2)
    if !isdir(overall_movie_folder_2)
        mkpath(overall_movie_folder_2)
    end

    λ_vals_to_plot=[]
    τ_0_values = [] 
    τ_tilde_bounds=[]   
    isirreducible_values=[]
    λ_transition=0
    for λ in lambda_values
        currlambdaforfilename=round(Int,λ*100)

        model = Model(Ipopt.Optimizer)
        # set_optimizer_attribute(model, "tol", 1e-8)
        set_optimizer_attribute(model, "print_level", 0)

        S, Skeyer, T, TG, TB, Tc = statematrices(N)
        ni, np, ns, nt = varioussizes(N)
        targetstates_good = [target_state + 1 for target_state ∈ TG]
        targetstates_bad = [target_state + 1 for target_state ∈ TB]
        targetstates = [targetstates_good; targetstates_bad]
        startstates = [start_state + 1 for start_state ∈ Tc]
        allstates = [startstates; targetstates_good; targetstates_bad]
        all_targetstates = vcat(targetstates_good, targetstates_bad)

        @variable(model, 0 <= P_[1:np] <= 1) 
        @variable(model, τ[1:ni]) 
        @objective(model, Min, τ[1])

        @expression(model, A, hitting_time_mod_give_A(Q_maker_original_mod(P_, N, λ, model, S, Skeyer), 
                      targetstates_good, targetstates_bad, allstates, λ, model))
        @expression(model, b, hitting_time_mod_give_b(Q_maker_original_mod(P_, N, λ, model, S, Skeyer), 
                      targetstates_good, targetstates_bad, allstates, λ, model))
        @constraint(model, A * τ == b)

        for i in 1:np
            set_start_value(P_[i],initialPval)
        end
        for i in 1:ns
            set_start_value(τ[i], initialtauval)
        end

        optimize!(model)
        P_opt = value.(P_)
        tau_opt = value.(τ)

        if (termination_status(model) != MOI.LOCALLY_SOLVED)
            println("SKIPPED ->",λ," ", termination_status(model))
           continue
        end
        P_opt_ = P_opt .* (abs.(P_opt) .>= 1e-8)
        P_opt_ .= min.(P_opt_, 1.0)
        Q_opt = Q_maker_using_M(P_opt_, N, λ, S, Skeyer)

        println(is_irreducible(Q_opt))

        if (!is_irreducible(Q_opt))
            println("SKIPPED ->",λ, ",",tau_opt[1]," : The solution found leads to a redducible rate matrix")
            initialtauval=0.0
            initialPval=1
            continue
        end
        lambda_str = replace(string(λ), "." => "_")
        sub_sub_folder_ = @sprintf("lambda_%s", lambda_str)
        overall_sub_sub_folder=joinpath(overall_folder,sub_sub_folder_)
        if !isdir(overall_sub_sub_folder)
            mkpath(overall_sub_sub_folder)
        end
        initialPval=1
        initialtauval=tau_opt[1]
        tau_opt_tilde = tau_opt[startstates]
        upper_bound=maximum(tau_opt_tilde)
        push!(λ_vals_to_plot,λ)
        push!(τ_0_values, tau_opt[1])
        push!(τ_tilde_bounds, upper_bound)
        push!(isirreducible_values,is_irreducible(Q_opt))
        println(λ," ",termination_status(model)," ", tau_opt[1])


        Q_filename = generate_filename(overall_sub_sub_folder,"Q_matrix_heatmap")
        plot_Q_with_colored_yticks(Q_opt, N, all_targetstates, Q_filename,λ)

        initial_state = 1
        T = 100.0
        times, states = simulate_ctmc(Q_opt, initial_state, T)
        
        ctmc_simulation_filename = generate_filename(overall_sub_sub_folder,"single_ctmc_simulation")
        plot_ctmc_our_problem(times, states, T, N, ctmc_simulation_filename,λ)

        initial_state = 1
        T = 7000.0

        longtime_heatmap_simulation_filename = generate_filename(overall_sub_sub_folder,"multiple_ctmc_simulation_heatmap_longtime")
        plot_ctmc_our_problem_multi(Q_opt, initial_state, T, N, num_simulations, longtime_heatmap_simulation_filename,λ)


        invariant_heatmap_simulation_filename = generate_filename(overall_sub_sub_folder,"invariant_ctmc_heatmap")
        what_is_the_current_status=plot_ctmc_invar_distn_our_problem(Q_opt,  N, invariant_heatmap_simulation_filename, λ)

        if(what_is_the_current_status)
            λ_transition=λ
        end

        frame_filename = generate_filename(overall_movie_folder, @sprintf("final_frame_lambda_%s",lambda_str))
        source_path = longtime_heatmap_simulation_filename*".png"
        cp(source_path, frame_filename*".png"; force=true)

        frame_filename = generate_filename(overall_movie_folder_2, @sprintf("final_frame_lambda_%s",lambda_str))
        source_path = invariant_heatmap_simulation_filename*".png"
        cp(source_path, frame_filename*".png"; force=true)
    end
    transition_column = [λ == λ_transition for λ in λ_vals_to_plot]
    title_name=@sprintf("Optimal τ₀, N=%d",N)
    display(plot(λ_vals_to_plot, τ_0_values, xlabel="λ", ylabel="Optimal τ₀", lw=2, legend=false, title=title_name))
    plot_file_name = generate_filename(overall_folder,"plot_of_optimal_tau_0_vs_lambda")
    savefig(plot_file_name*".png")
    savefig(plot_file_name*".svg")

    title_name=@sprintf("Upper Bound on τ̃ , N=%d",N)
    display(plot(λ_vals_to_plot, τ_tilde_bounds, xlabel="λ", ylabel="Upper Bound on τ̃", lw=2, legend=false, title=title_name))
    plot_file_name = generate_filename(overall_folder,"plot_of_tau_tilde_upper_bound_vs_lambda")
    savefig(plot_file_name*".png")
    savefig(plot_file_name*".svg")

    df = DataFrame(Lambda_Values = λ_vals_to_plot, Tau_0_Values = τ_0_values, IsIrreducible = isirreducible_values,Tau_tilde_bounds=τ_tilde_bounds,IsTransitionPoint = transition_column)
    csv_filename=generate_filename(overall_folder, @sprintf("lambda_tau_values"))
    CSV.write(csv_filename*".csv", df)

    # movie1 = generate_filename(overall_folder, @sprintf("longtimeruns_movie_N_%d",N))
    # movie2 = generate_filename(overall_folder, @sprintf("invariantdistr_movie_N_%d",N))

    return λ_vals_to_plot, τ_0_values, isirreducible_values
end

function Q_maker_original_mod(P,N::Int64,λ::Float64, model,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    Q=@expression(model, zeros(AffExpr, ni, ni)) 
    for (u, u_) in Iterators.product(values(S), values(S))
        if u == u_
            continue
        end
        i = Skeyer[u]
        j = Skeyer[u_]
        flag = 0
        ua, ub, sa, sb = u
        ua_, ub_, sa_, sb_ = u_

        tempk = 0

        if ua_==ua+1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sa + ua + 1
        elseif ua_==ua-1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sa * N + ua + 2 * N 
        elseif ub_==ub+1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sb + ub + 1
        elseif ub_==ub-1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sb * N + ub + 2 * N 
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub + 1
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 2
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua + 1
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 2
        end
        if flag == 1
            Q[i+1, j+1] = P[tempk]
        end
    end
    for i in 1:ni
        qi = sum(Q[i, :])
        Q[i,i] = -qi
    end
    return Q
end

function M_maker(N::Int64,λ::Float64,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    M=zeros(Int, ni, ni, np);
    for (u, u_) in Iterators.product(values(S), values(S))
        if u == u_
            continue
        end

        i = Skeyer[u]
        j = Skeyer[u_]
        flag = 0
        ua, ub, sa, sb = u
        ua_, ub_, sa_, sb_ = u_

        tempk = 0

        if ua_==ua+1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sa + ua + 1
        elseif ua_==ua-1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sa * N + ua + 2 * N 
        elseif ub_==ub+1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sb + ub + 1
        elseif ub_==ub-1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sb * N + ub + 2 * N 
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub + 1
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 2
        elseif ua == ua_ && ub == ub_  && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua + 1
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 2
        end

        if flag == 1
            M[i+1, j+1, tempk] = flag
        end
    end
    for k in 1:np
        # Calculate the sum of each row for the kth slice of M (equivalent to M[:,:,k] in Python)
        M_i = [sum(M[:, :, k][i, :]) for i in 1:ni]

        for u in values(S)
            i = Skeyer[u]
            M[i+1, i+1, k] = -M_i[i+1]
        end
    end
    
    return M
end

function Q_maker_using_M(P,N::Int64,λ::Float64,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    M=M_maker(N::Int64,λ::Float64,S,Skeyer)
    Q = reduce((x, y) -> x + y, [P[k] * M[:, :, k] for k in 1:np])
    return Q
end

function M_maker_mod(N::Int64,λ::Float64, model,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    M=@expression(model, zeros(AffExpr, ni, ni, np)) 
    for (u, u_) in Iterators.product(values(S), values(S))
        if u == u_
            continue
        end

        i = Skeyer[u]
        j = Skeyer[u_]
        flag = 0
        ua, ub, sa, sb = u
        ua_, ub_, sa_, sb_ = u_

        tempk = 0

        if ua_==ua+1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sa + ua + 1
        elseif ua_==ua-1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sa * N + ua + 2 * N 
        elseif ub_==ub+1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sb + ub + 1
        elseif ub_==ub-1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sb * N + ub + 2 * N 
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub + 1
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 2
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua + 1
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 2
        end

        if flag == 1
            M[i+1, j+1, tempk] = flag
        end
    end
    for k in 1:np
        M_i = [sum(M[:, :, k][i, :]) for i in 1:ni]

        for u in values(S)
            i = Skeyer[u]
            M[i+1, i+1, k] = -M_i[i+1]
        end
    end
    
    return M
end

function Q_maker_using_M_mod(P,N::Int64,λ::Float64,model,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    M=M_maker_mod(N,λ,model,S,Skeyer)
    Q = reduce((x, y) -> x + y, [P[k] * M[:, :, k] for k in 1:np])
    return Q
end

function Q_maker_tilde_mod(P,N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    Q=Q_maker_original_mod(P,N,λ, model,S,Skeyer)
    R = zeros(ns, ni)
    for i in 1:ns
        R[i,Tc[i]+1] = 1
    end
    Qtilde = R * Q * R'
    return Qtilde
end

function M_maker_tilde(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    M=M_maker(N,λ,S,Skeyer)
    R = zeros(ns, ni)
    for i in 1:ns
        R[i,Tc[i]+1] = 1
    end
    
    Mtilde=zeros(ns, ns, np)
    for k in 1:np
        Mtilde[:, :, k] = R * M[:, :, k] * R'
    end

    return Mtilde
end

function M_maker_tilde_mod(N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    M=M_maker_mod(N,λ,model,S,Skeyer)
    R = zeros(ns, ni)
    for i in 1:ns
        R[i,Tc[i]+1] = 1
    end
    Mtilde=@expression(model, zeros(AffExpr, ns, ns, np)) 
    for k in 1:np
        Mtilde[:, :, k] = R * M[:, :, k] * R'
    end

    return Mtilde
end

function A_maker(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    Mtilde=M_maker_tilde(N,λ,S,Skeyer,T,TG,TB,Tc)
    Ai = zeros(ns, np, ns)
    for i in 1:ns
        Ai[:,:,i]= Mtilde[i,:,:]   
    end
    return Ai
end

function A_maker_mod(N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    Mtilde=M_maker_tilde_mod(N,λ, model,S,Skeyer,T,TG,TB,Tc)
    Ai = @expression(model, zeros(AffExpr, ns, np, ns)) 
    for i in 1:ns
        Ai[:,:,i]= Mtilde[i,:,:]   
    end
    return Ai
end

function alpha_maker(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    alpha=zeros(np, ns)
    M=M_maker(N,λ,S,Skeyer)
    for i in 1:ns
        Si=Tc[i]
        for j in 1:ni
            for k in 1:np
                if (j-1 ∈ TB)
                    alpha[k,i]+=M[Si+1,j,k]
                end
                
            end
        end
    end
    return alpha
end

function alpha_maker_mod(N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    alpha=@expression(model, zeros(AffExpr, np, ns))
    M=M_maker_mod(N,λ,model,S,Skeyer)
    for i in 1:ns
        Si=Tc[i]
        for j in 1:ni
            for k in 1:np
                if (j-1 ∈ TB)
                    alpha[k,i]+=M[Si+1,j,k]
                end
                
            end
        end
    end
    return alpha
end

function D_maker(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    Di=zeros(nt, nt, ns)

    alpha=alpha_maker(N,λ,S,Skeyer,T,TG,TB,Tc)
    Ai=A_maker(N,λ,S,Skeyer,T,TG,TB,Tc)
    
    for i in 1:ns
        Di[np+1:np+ns,1:np,i]=Ai[:,:,i]
        Di[1:np,nt,i]=λ*alpha[:,i]'
        Di[nt,nt,i]=1
    end
    D_i=copy(Di)
    for i in 1:ns
        D_i[ :, :,i] = (Di[ :, :,i] + Di[ :, :,i]') / 2
    end
    return D_i
end

function E_maker(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    Ei=zeros(nt, nt, nt)

    alpha=alpha_maker(N,λ,S,Skeyer,T,TG,TB,Tc)
    Ai=A_maker(N,λ,S,Skeyer,T,TG,TB,Tc)
    
    for i in 1:nt
        Ei[i,nt,i]=1
    end
    
    E_i=copy(Ei)
    
    for i in 1:nt
        E_i[ :, :,i] = (Ei[ :, :,i] + Ei[ :, :,i]') / 2
    end
    return E_i
end

function C_maker(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    C_i=zeros(nt, nt, np)
    
    for i in 1:np
        C_i[i,i,i]=1
    end
    return C_i
end

function F_maker(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    Fij=zeros(nt, nt, np, np)
    for i in 1:np
        for j in 1:np
            Fij[i,j,i,j]=1
        end
    end
    Fiji=copy(Fij)
    for i in 1:np
        for j in 1:np
            Fiji[ :, :,i,j] = (Fij[ :, :,i,j] + Fij[ :, :,i,j]') / 2
        end
    end
    print(size(Fiji))
    return Fiji
end


struct Node
    lb::Float64
    bounds::Vector{Tuple{Float64, Float64}}
    depth::Int
end

function solve_relaxation(node::Node, S0, Ci, Di,Ei, np,ns,nt)
    L_node = [node.bounds[i][1] for i in 1:length(node.bounds)]
    U_node = [node.bounds[i][2] for i in 1:length(node.bounds)]
    
    model = Model(COSMO.Optimizer)
    # set_optimizer_attribute(model, "max_iter", 100000)
    set_optimizer_attribute(model, "eps_abs", 1e-4)
    set_silent(model) 
    
    @variable(model, X[1:nt, 1:nt], Symmetric)#Sym
    @objective(model, Min, tr(X * S0))
    @constraint(model, X in PSDCone())#PSD

    @constraint(model, X .>= 0) #DNN
    
    for k in 1:ns
        @constraint(model, tr(X * Di[:,:,k]) == 0.0)
    end

    for i=1:nt-1
        for j=1:np
            @constraint(model, X[i,j] <= X[i,nt])
        end
    end
    for i in 1:nt
        for j in 1:nt
            @constraint(model, X[i,j] >= L_node[i] * L_node[j])
            @constraint(model, X[i,j] <= U_node[i] * U_node[j])
        end
    end
    @constraint(model, X[nt,nt] == 1)
    
    optimize!(model)
    
    status = termination_status(model)
    if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
        lb = objective_value(model)
        X_val = value.(X)
        # eigenvalues, eigenvectors = eigen(X_val)
        # eigenvalues = max.(eigenvalues, 0)
        # Z_relaxed = eigenvectors[:, eigenvalues .> 1e-6] * sqrt.(eigenvalues[eigenvalues .> 1e-6])
        # if size(Z_relaxed, 2) > 1
        #     Z_relaxed = vec(sum(Z_relaxed; dims=2))
        # else
        #     Z_relaxed = vec(Z_relaxed)
        # end
        # Z_relaxed = X_val[:, nt]
        return status, lb, X_val
    else
        return status, Inf, nothing
    end
end

# function is_feasible(Z::Vector{Float64}, bounds::Vector{Tuple{Float64, Float64}}, Ci, Di, Ei, n_p, n_s, n_t)
#     L_node = [node.bounds[i][1] for i in 1:length(node.bounds)]
#     U_node = [node.bounds[i][2] for i in 1:length(node.bounds)]
    
#     model = Model(COSMO.Optimizer)
#     set_optimizer_attribute(model, "max_iter", 3000)
#     set_optimizer_attribute(model, "eps_abs", 1e-4)
#     # set_silent(model) 
    
#     @variable(model, X[1:nt, 1:nt], Symmetric)#Sym
#     @objective(model, Min, tr(X * S0))
#     @constraint(model, X in PSDCone())#PSD

#     # for i in 1:nt
#     #     @constraint(model, X[i,i] >= L_node[i] * L_node[i])
#     #     @constraint(model, X[i,i] <= U_node[i] * U_node[i])
#     # end
#     @constraint(model, X .>= 0) #DNN
#     # for k in 1:np
#     #     @constraint(model, tr(X * Ci[:,:,k]) <=1.0)
#     # end
    
#     for k in 1:ns
#         @constraint(model, tr(X * Di[:,:,k]) == 0.0)
#     end

#     # for k in 1:nt
#     #     @constraint(model, tr(X * Ei[:,:,k]) >= 0.0)
#     # end

#     for i=1:nt-1
#         for j=1:np
#             @constraint(model, X[i,j] <= X[i,nt])
#         end
#     end
#     for i in 1:nt
#         for j in 1:nt
#             @constraint(model, X[i,j] >= L_node[i] * L_node[j])
#             @constraint(model, X[i,j] <= U_node[i] * U_node[j])
#         end
#     end
#     @constraint(model, X[nt,nt] == 1)
    
#     optimize!(model)
    
#     status = termination_status(model)
#     if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
#         lb = objective_value(model)
#         X_val = value.(X)
#         # eigenvalues, eigenvectors = eigen(X_val)
#         # eigenvalues = max.(eigenvalues, 0)
#         # Z_relaxed = eigenvectors[:, eigenvalues .> 1e-6] * sqrt.(eigenvalues[eigenvalues .> 1e-6])
#         # if size(Z_relaxed, 2) > 1
#         #     Z_relaxed = vec(sum(Z_relaxed; dims=2))
#         # else
#         #     Z_relaxed = vec(Z_relaxed)
#         # end
#         # Z_relaxed = X_val[:, nt]
#         return status, lb, X_val
#     else
#         # Infeasible or unbounded
#         return status, Inf, nothing
#     end
# end

function select_branching_variable(bounds::Vector{Tuple{Float64, Float64}})
    widths = [bounds[i][2] - bounds[i][1] for i in 1:length(bounds)]
    return argmax(widths)
end


