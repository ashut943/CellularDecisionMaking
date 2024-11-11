using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, CSV, DataFrames, LsqFit, LaTeXStrings
include("utils.jl")
include("two_cell_functions_ctmc.jl")

Nvals = [2, 3, 4, 5, 6, 7]
lambda_start_now = 0.0
lambda_end_now = 300.0
start_val = 10

plot(
    # title="λ vs Optimal τ₀ for Various N",
    xlabel=L"\lambda", ylabel=L"\tau^*_0",
    lw=2, framestyle=:box
)

palette = [RGB(0.3, 0.55, 0.75), RGB(0.8, 0.47, 0.44), RGB(0.35, 0.7, 0.5), RGB(0.5, 0.6, 0.7), RGB(0.6, 0.5, 0.8), RGB(0.7, 0.4, 0.4), RGB(0.4, 0.7, 0.9)]

for (i, N) in enumerate(Nvals)
    lambda_start_str = replace(string(lambda_start_now), "." => "_")
    lambda_end_str = replace(string(lambda_end_now), "." => "_")
    main_folder_curr = @sprintf("Simulations_N_%d_Interior_Point_Methods", N)
    main_subfolder_curr = @sprintf("simulation_parameter_results_N_%d_lambdastart_%s_lambdaend_%s", N, lambda_start_str, lambda_end_str)
    overall_subfolder_curr = joinpath(main_folder_curr, main_subfolder_curr)
    curr_csv_filename = generate_filename(overall_subfolder_curr, @sprintf("lambda_tau_values"))

    if isfile(curr_csv_filename * ".csv")
        df = CSV.read(curr_csv_filename * ".csv", DataFrame)
        λ_vals_to_plot_ = df.Lambda_Values
        τ_0_values_ = df.Tau_0_Values
        valid_indices = findall(x -> x >= start_val, λ_vals_to_plot_)
        λ_vals_to_plot_ = λ_vals_to_plot_[valid_indices]
        τ_0_values_ = τ_0_values_[valid_indices]
        valid_indices = findall(x -> x > 0, λ_vals_to_plot_) ∩ findall(x -> x > 0, τ_0_values_)
        λ_vals_to_plot_filtered = λ_vals_to_plot_[valid_indices]
        τ_0_values_filtered = τ_0_values_[valid_indices]

        log_λ_vals = log.(λ_vals_to_plot_filtered)
        log_τ_vals = log.(τ_0_values_filtered)

        model(x, p) = p[1] .+ p[2] .* x
        initial_params = [0.0, 1.0]
        fit = LsqFit.curve_fit(model, log_λ_vals, log_τ_vals, initial_params)
        slope = fit.param[2]
        intercept = fit.param[1]
        println("Gradient (slope) of log-log plot for N = $N: ", slope)

        λ_vals_fit = range(start_val, stop=300, length=100)
        τ_0_vals_fit = exp.(intercept .+ slope .* log.(λ_vals_fit))

        scatter!(λ_vals_to_plot_filtered, τ_0_values_filtered, color=palette[i], marker=:circle, markersize=2, label="N = $(N)")
        plot!(λ_vals_fit, τ_0_vals_fit, lw=2, linestyle=:dash, color=palette[i], label="")
    else
        println("File not found: $curr_csv_filename")
    end
end

# Save the plot
plot!()
savefig("lambda_vs_optimal_tau0_upper_bounds_N2to7.png")
