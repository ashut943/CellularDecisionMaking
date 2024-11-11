using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, CSV, DataFrames, LsqFit
include("utils.jl")
include("two_cell_functions_ctmc.jl")


Nvals=[2,3,4,5,6]
lambda_start_now=0.0
lambda_end_now=300.0

model(λ, p) = p[1] * log.(λ .+ p[2]) .+ p[3]  # p[1]=A, p[2]=B, p[3]=C
model(λ, p) = (p[1] * λ .+ p[2]).^p[3] .+ p[4]  # p[1]=A, p[2]=B, p[3]=C, p[4]=D
plot(title="λ vs optimal τ₀ for Various N", xlabel="λ", ylabel="optimal τ₀")
for N in Nvals
    lambda_start_str = replace(string(lambda_start_now), "." => "_")
    lambda_end_str=lambda_str = replace(string(lambda_end_now), "." => "_")
    main_folder_curr = @sprintf("Simulations_N_%d_Interior_Point_Methods", N)
    main_subfolder_curr= sub_folder = @sprintf("simulation_parameter_results_N_%d_lambdastart_%s_lambdaend_%s", N, lambda_start_str,lambda_end_str)
    overall_subfolder_curr=joinpath(main_folder_curr,main_subfolder_curr)
    curr_csv_filename=generate_filename(overall_subfolder_curr, @sprintf("lambda_tau_values"))
    if isfile(curr_csv_filename*".csv")
        df = CSV.read(curr_csv_filename*".csv", DataFrame)
        λ_vals_to_plot_ = df.Lambda_Values
        τ_0_values_ = df.Tau_0_Values
        isirreducible_=df.IsIrreducible
        # Plot λ vs τ_0 for the current N with a label
        plot!(λ_vals_to_plot_, τ_0_values_, label="N = $N")
        # Initial guess for parameters [A, B, C, D]
        p0 = [1.0, 1.0, 1.0]
        p0 = [1.0, 1.0, 1.0, 1.0]

        # Perform the fitting
        # fit = curve_fit(model, λ_vals_to_plot_, τ_0_values_, p0)
        # # A, B, C = fit.param
        # A, B, C, D = fit.param
        # # println(N," ",A," ",B," ",C)
        # println(N," ",A," ",B," ",C," ",D)

        # # Generate fitted values
        # τ_0_fitted = model(λ_vals_to_plot_, fit.param)
        
        # # Plot fitted curve
        # plot!(λ_vals_to_plot_, τ_0_fitted, label="Fit N = $N)", lw=2, ls=:dash)    
    else
        println("File not found: $curr_csv_filename")
    end   
end
plot!()
# λ_vals_to_plot, τ_0_values, isirreducible=run_pipeline_for_various_lambda(N, lambda_values, num_simulations,initialPval,initialtauval)
plot(title="λ vs optimal τ₀ for Various N (Log-Log Scale)", xlabel="log(λ)", ylabel="log(optimal τ₀)")
for N in Nvals
    lambda_start_str = replace(string(lambda_start_now), "." => "_")
    lambda_end_str=lambda_str = replace(string(lambda_end_now), "." => "_")
    main_folder_curr = @sprintf("Simulations_N_%d_Interior_Point_Methods", N)
    main_subfolder_curr = @sprintf("simulation_parameter_results_N_%d_lambdastart_%s_lambdaend_%s", N, lambda_start_str, lambda_end_str)
    overall_subfolder_curr = joinpath(main_folder_curr, main_subfolder_curr)
    curr_csv_filename = generate_filename(overall_subfolder_curr, "lambda_tau_values")
    
    if isfile(curr_csv_filename * ".csv")
        df = CSV.read(curr_csv_filename * ".csv", DataFrame)
        λ_vals_to_plot_ = df.Lambda_Values
        τ_0_values_ = df.Tau_0_Values
        
        # Filter out non-positive values for log-log plot
        valid_indices = findall(x -> x > 0, λ_vals_to_plot_) ∩ findall(x -> x > 0, τ_0_values_)
        λ_vals_to_plot_filtered = λ_vals_to_plot_[valid_indices]
        τ_0_values_filtered = τ_0_values_[valid_indices]
        
        # Log-log scale plot for the current N
        plot!(log.(λ_vals_to_plot_filtered), log.(τ_0_values_filtered), label="N = $N")
    else
        println("File not found: $curr_csv_filename")
    end
end
plot!()

# plot(title="λ vs optimal τ₀ for Various N (exp-Linear Scale)", xlabel="λ", ylabel="exp(optimal τ₀)")
# for N in Nvals
#     lambda_start_str = replace(string(lambda_start_now), "." => "_")
#     lambda_end_str=lambda_str = replace(string(lambda_end_now), "." => "_")
#     main_folder_curr = @sprintf("Simulations_N_%d_Interior_Point_Methods", N)
#     main_subfolder_curr = @sprintf("simulation_parameter_results_N_%d_lambdastart_%s_lambdaend_%s", N, lambda_start_str, lambda_end_str)
#     overall_subfolder_curr = joinpath(main_folder_curr, main_subfolder_curr)
#     curr_csv_filename = generate_filename(overall_subfolder_curr, "lambda_tau_values")
    
#     if isfile(curr_csv_filename * ".csv")
#         df = CSV.read(curr_csv_filename * ".csv", DataFrame)
#         λ_vals_to_plot_ = df.Lambda_Values
#         τ_0_values_ = df.Tau_0_Values
        
#         # Filter out non-positive values for log-log plot
#         valid_indices = findall(x -> x > 0, λ_vals_to_plot_) ∩ findall(x -> x > 0, τ_0_values_)
#         λ_vals_to_plot_filtered = λ_vals_to_plot_[valid_indices]
#         τ_0_values_filtered = τ_0_values_[valid_indices]
        
#         # Log-log scale plot for the current N
#         plot!((λ_vals_to_plot_filtered), exp.(τ_0_values_filtered), label="N = $N")
#     else
#         println("File not found: $curr_csv_filename")
#     end
# end
# plot!()