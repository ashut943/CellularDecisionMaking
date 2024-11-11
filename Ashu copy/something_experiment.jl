using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, DataStructures, MathOptInterface
const MOI = MathOptInterface
include("utils.jl")
include("two_cell_functions_ctmc.jl")

# Define parameters
N = 3
λ_values = [1.0, 2.0]  # Different lambda values to be tested

# Initialize storage for data from each lambda
eigenvalues_data_λ = Dict()
negative_eigenvalues_data_λ = Dict()
indices_λ = Dict()
negative_indices_λ = Dict()
diagonalizable_λ = Dict()

for λ in λ_values
    # Generate matrices and dimensions based on lambda
    S, Skeyer, T, TG, TB, Tc = statematrices(N)
    ni, np, ns, nt = varioussizes(N)
    Di = D_maker(N, λ, S, Skeyer, T, TG, TB, Tc)
    
    # Initialize storage for this lambda's results
    eigenvalues_data = []
    indices = []
    negative_eigenvalues_data = []
    negative_indices = []
    diagonalizable = []
    
    # Loop through matrices and analyze eigenvalues
    for i in 1:ns
        local D_i = Di[:, :, i]
        eig = eigen(D_i)
        eigenvector_matrix = eig.vectors
        real_eigenvalues = real.(eig.values)
        
        append!(eigenvalues_data, real_eigenvalues)
        append!(indices, fill(i, length(real_eigenvalues)))
        
        # Check for diagonalizability
        is_diagonalizable = rank(eigenvector_matrix) == size(D_i, 1)
        push!(diagonalizable, is_diagonalizable)
        
        # Store negated positive eigenvalues
        for val in real_eigenvalues
            if val > 0
                push!(negative_eigenvalues_data, -val)
                push!(negative_indices, i)
            end
        end
    end
    
    # Store the results for this lambda
    eigenvalues_data_λ[λ] = eigenvalues_data
    indices_λ[λ] = indices
    negative_eigenvalues_data_λ[λ] = negative_eigenvalues_data
    negative_indices_λ[λ] = negative_indices
    diagonalizable_λ[λ] = diagonalizable
end

# Plotting the results
plot(title="Eigenvalues of D_i Matrices for Different λ Values", xlabel="Index i", ylabel="Real Eigenvalues")

# Plot each lambda's eigenvalues and negated positive values
for λ in λ_values
    println(λ)
    display(scatter!(indices_λ[λ], eigenvalues_data_λ[λ], label="Eigenvalues (λ = $λ)", legend=:topright))
    
    # display(scatter!(negative_indices_λ[λ], negative_eigenvalues_data_λ[λ], marker=:x, color=:red, label="Negated Positive Eigenvalues (λ = $λ)"))
end

