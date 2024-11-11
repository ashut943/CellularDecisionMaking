using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, DataStructures, MathOptInterface
const MOI = MathOptInterface
include("utils.jl")
include("two_cell_functions_ctmc.jl")

N=3
λ=30.0

S,Skeyer,T,TG,TB,Tc=statematrices(N);
ni,np,ns,nt=varioussizes(N)

targetstates_good=[target_state+1 for target_state ∈ TG];
targetstates_bad=[target_state+1 for target_state ∈ TB];
targetstates=[targetstates_good;targetstates_bad]
startstates=[start_state+1 for start_state ∈ Tc];
allstates=[startstates;targetstates_good; targetstates_bad]
all_targetstates = vcat(targetstates_good, targetstates_bad)

Di=D_maker(N,λ,S,Skeyer,T,TG,TB,Tc);
Ei=E_maker(N,λ,S,Skeyer,T,TG,TB,Tc);
S0=Ei[:,:,np+1];
Ci=C_maker(N,λ,S,Skeyer,T,TG,TB,Tc);

# for i=1:ns
#     global D_i=Di[:,:,i]
#     println("====================")
#     println(i)
#     # heatmap_plot=heatmap(D_i, c=:coolwarm, title="Transition Rate Matrix Heatmap")
#     # display(heatmap_plot)
#     eig = eigen(D_i)
#     eigenvalues = eig.values
#     eigenvectors = eig.vectors
#     real_parts = real.(eigenvalues)
#     imag_parts = imag.(eigenvalues)
#     # Set tolerance (for example, 1e-8)
#     tolerance = 1e-8
    
#     println(eigenvalues)
#     # display(scatter(real_parts, imag_parts, xlabel="Real Part", ylabel="Imaginary Part", title="Argand Diagram", legend=false, grid=true))
#     # Check for "essential zeros"
#     # essential_zeros = [isapprox(x, 0.0, atol=tolerance) for x in eigenvalues]

#     # To find the indices of values that are essentially zero
#     # zero_indices = findall(x -> isapprox(x, 0.0, atol=tolerance), eigenvalues)
#     # Display the eigenvalues
#     # println("The eigenvalues of Di are: ", zero_indices)
# end

# eigenvalues_data = []
# indices = []
# diagonalizable = []

# for i in 1:ns
#     local D_i = Di[:, :, i]
#     eig = eigen(D_i)
#     eigenvector_matrix = eig.vectors
#     real_eigenvalues = real.(eig.values)
    
#     append!(eigenvalues_data, real_eigenvalues)
#     append!(indices, fill(i, length(real_eigenvalues))) 
#     is_diagonalizable = rank(eigenvector_matrix) == size(D_i, 1)
#     push!(diagonalizable, is_diagonalizable)
    
#     println("D_$i is $(is_diagonalizable ? "" : "not ")diagonalizable")
# end


# scatter(indices, eigenvalues_data, xlabel="Index i", ylabel="Real Eigenvalues", 
#         title="Eigenvalues of D_i Matrices", legend=false, grid=true)
eigenvalues_data = []
indices = []
negative_eigenvalues_data = []  # To store negated positive eigenvalues
negative_indices = []           # To store corresponding indices for negated values
diagonalizable = []

target_eigenvalue = 0.5 # Eigenvalue of interest
tolerance = 1e-3         # Tolerance for numerical matching
eigenvector_sets = []    # To store eigenvectors corresponding to the target eigenvalue


for i in 1:ns
    local D_i = Di[:, :, i]
    eig = eigen(D_i)
    eigenvector_matrix = eig.vectors
    eigenvectors = eig.vectors
    eigenvalues = real.(eig.values)
    real_eigenvalues = real.(eig.values)
    
    # Store real eigenvalues and corresponding index
    append!(eigenvalues_data, real_eigenvalues)
    append!(indices, fill(i, length(real_eigenvalues)))

    matching_vectors = [eigenvectors[:, j] for j in 1:length(eigenvalues) if abs(eigenvalues[j] - target_eigenvalue) < tolerance]

    
    # Check for diagonalizability
    is_diagonalizable = rank(eigenvector_matrix) == size(D_i, 1)
    push!(diagonalizable, is_diagonalizable)
    
    println("D_$i is $(is_diagonalizable ? "" : "not ")diagonalizable")
    
    # Find positive eigenvalues and store their negation
    for val in real_eigenvalues
        if val > 0
            push!(negative_eigenvalues_data, -val)
            push!(negative_indices, i)
        end
    end
    if !isempty(matching_vectors)
        push!(eigenvector_sets, matching_vectors[1])  # Store only the first matching eigenvector
    else
        push!(eigenvector_sets, nothing)  # For consistency, insert a placeholder if no match is found
    end
end

# Single scatter plot for all eigenvalues and negated positive values
scatter(indices, eigenvalues_data, xlabel="Index i", ylabel="Real Eigenvalues", 
        title="Eigenvalues of D_i Matrices", legend=false, grid=true, label="Eigenvalues")

# Overlay with negative of positive eigenvalues as red crosses
scatter!(negative_indices, negative_eigenvalues_data, marker=:x, color=:red, label="Negated Positive Eigenvalues")

println(eigenvector_sets)

# Convert the list of eigenvectors to a matrix for plotting
eigenvectors_matrix = hcat(eigenvector_sets...)  # Each eigenvector becomes a column

# Plot heatmap of eigenvectors associated with the target eigenvalue
heatmap(eigenvectors_matrix, color=:viridis, title="Eigenvectors associated with Eigenvalue $target_eigenvalue",
        xlabel="Matrix Index (D_i)", ylabel="Eigenvector Component", clims=(-1, 1))


# target_eigenvalue = 0 # Eigenvalue of interest
# tolerance = 1e-3         # Tolerance for numerical matching
# eigenvector_sets = []    # To store eigenvectors corresponding to the target eigenvalue

# # Collect eigenvectors associated with the target eigenvalue across all D_i matrices
# for i in 1:ns
#     local D_i = Di[:, :, i]
#     eig = eigen(D_i)
#     eigenvalues = real.(eig.values)
#     eigenvectors = eig.vectors
    
#     # Find eigenvectors corresponding to the target eigenvalue (within tolerance)
#     matching_vectors = [eigenvectors[:, j] for j in 1:length(eigenvalues) if abs(eigenvalues[j] - target_eigenvalue) < tolerance]
    
#     # If there is at least one matching eigenvector, take the first one (or adjust as needed)
#     if !isempty(matching_vectors)
#         push!(eigenvector_sets, matching_vectors[1])  # Store only the first matching eigenvector
#     else
#         push!(eigenvector_sets, nothing)  # For consistency, insert a placeholder if no match is found
#     end
# end

# # Calculate cosine similarity between eigenvectors for each pair of D_i matrices
# function cosine_similarity(v1, v2)
#     return dot(v1, v2) / (norm(v1) * norm(v2))
# end

# similarity_matrix = fill(NaN, ns, ns)  # Matrix to store similarities (initialize with NaNs)

# for i in 1:ns
#     for j in i:ns
#         if eigenvector_sets[i] ≠ nothing && eigenvector_sets[j] != nothing
#             similarity_matrix[i, j] = cosine_similarity(eigenvector_sets[i], eigenvector_sets[j])
#             similarity_matrix[j, i] = similarity_matrix[i, j]  # Symmetric matrix
#         end
#     end
# end

# # Plot heatmap of cosine similarities
# display(heatmap(similarity_matrix, color=:viridis, title="Cosine Similarity of Eigenvectors",
#         xlabel="Matrix Index (D_i)", ylabel="Matrix Index (D_j)", clims=(0, 1))
# )