using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, DataStructures, MathOptInterface
const MOI = MathOptInterface
include("utils.jl")
include("two_cell_functions_ctmc.jl")

N=3
λ=3.0

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

target_eigenvalue = 0  # Eigenvalue of interest
tolerance = 1e-4       # Tolerance for numerical matching
eigenvectors_matrix = [] # Matrix to store eigenvectors as columns for each D_i
zero_threshold=1e-5

# Collect eigenvectors associated with the target eigenvalue across all D_i matrices
for i in 1:ns
    local D_i = Di[:, :, i]
    eig = eigen(D_i)
    eigenvalues = real.(eig.values)
    eigenvectors = eig.vectors
    
    # Find eigenvectors corresponding to the target eigenvalue (within tolerance)
    matching_vectors = [eigenvectors[:, j] for j in 1:length(eigenvalues) if abs(eigenvalues[j] - target_eigenvalue) < tolerance]
    
    # If there is at least one matching eigenvector, take the first one (or adjust as needed)
    if !isempty(matching_vectors)
        vector = matching_vectors[1]
        vector = [abs(val) < zero_threshold ? 0.0 : val for val in vector]
        push!(eigenvectors_matrix, vector)  # Store only the first matching eigenvector
    else
        # If no eigenvector matches, fill with zeros or NaNs for that matrix (for consistency)
        push!(eigenvectors_matrix, fill(NaN, size(D_i, 1)))
    end
end

# Convert the list of eigenvectors to a matrix for plotting
eigenvectors_matrix = hcat(eigenvectors_matrix...)  # Each eigenvector becomes a column

# Plot heatmap of eigenvectors associated with the target eigenvalue
heatmap(eigenvectors_matrix, color=:matter, title="Eigenvectors associated with Eigenvalue $target_eigenvalue",
        xlabel="Matrix Index (D_i)", ylabel="Eigenvector Component", clims=(-1, 1))
