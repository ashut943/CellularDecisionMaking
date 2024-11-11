using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, CSV, DataFrames, LsqFit, LaTeXStrings,ProgressMeter, VideoIO
include("utils.jl")
include("two_cell_functions_ctmc.jl")

N_vals=[2,3,4,5,6,7]
lambda_start_now = 0.0
lambda_end_now = 300.0
start_val = 10

main_folder = @sprintf("Simulations_N_%d_Interior_Point_Methods", N)


lambda_start = lambda_start_now
lambda_start_str = replace(string(lambda_start), "." => "_")
lambda_end = lambda_end_now
lambda_end_str=lambda_str = replace(string(lambda_end), "." => "_")


for N in N_vals
    global main_folder = @sprintf("Simulations_N_%d_Interior_Point_Methods", N)
    sub_folder = @sprintf("simulation_parameter_results_N_%d_lambdastart_%s_lambdaend_%s", N, lambda_start_str,lambda_end_str)
    overall_folder=joinpath(main_folder,sub_folder)
    movie_folder = @sprintf("movie_folder")
    overall_movie_folder=joinpath(overall_folder,movie_folder)
    movie_folder_2 = @sprintf("movie_folder_2")
    overall_movie_folder_2=joinpath(overall_folder,movie_folder_2)

    create_movie(overall_movie_folder, joinpath(overall_folder, "longtime_heatmap.mp4"), frame_rate=10)
    create_movie(overall_movie_folder_2, joinpath(overall_folder, "invar_distn.mp4"), frame_rate=10)
end