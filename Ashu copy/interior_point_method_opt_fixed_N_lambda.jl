using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, CSV, DataFrames, VideoIO
include("utils.jl")
include("two_cell_functions_ctmc.jl")
N = 2
main_folder = @sprintf("Simulations_N_%d_Interior_Point_Methods", N)

M=300
lambda_values = collect(range(0, stop=300.0, length=M+1)) 
num_simulations = 100 
initialPval=1.0
initialtauval=0.0

λ_vals_to_plot, τ_0_values, isirreducible=run_pipeline_for_various_lambda(N, lambda_values, num_simulations,initialPval,initialtauval)
