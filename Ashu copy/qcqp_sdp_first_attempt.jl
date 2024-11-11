using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO
include("utils.jl")
include("two_cell_functions_ctmc.jl")

N=5
λ=1.0

# model = Model(SCS.Optimizer)
model = Model(COSMO.Optimizer)

S,Skeyer,T,TG,TB,Tc=statematrices(N);
ni,np,ns,nt=varioussizes(N)

targetstates_good=[target_state+1 for target_state ∈ TG];
targetstates_bad=[target_state+1 for target_state ∈ TB];
targetstates=[targetstates_good;targetstates_bad]
startstates=[start_state+1 for start_state ∈ Tc];
allstates=[startstates;targetstates_good; targetstates_bad]

Di=D_maker(N,λ,S,Skeyer,T,TG,TB,Tc);
Ei=E_maker(N,λ,S,Skeyer,T,TG,TB,Tc);
S0=Ei[:,:,np+1];
Ci=C_maker(N,λ,S,Skeyer,T,TG,TB,Tc);

@variable(model, X[1:nt, 1:nt], Symmetric)#Sym
@objective(model, Min, tr(X * S0))
@constraint(model, X in PSDCone())#PSD
# @constraint(model, X .>= 0) #DNN
for i=1:np
    @constraint(model, tr(X * Ci[:,:,i]) <=1.0)
end
for i=1:ns
    @constraint(model, tr(X * Di[:,:,i]) == 0.0)
end
@constraint(model, tr(X * Ei[:,:,nt]) == 1)
for i=1:nt-1
    @constraint(model, tr(X * Ei[:,:,i]) >= 0)
end
# @constraint(model, X[nt,nt] == 1)

optimize!(model)
solution = value.(X)
tau_0=objective_value(model)
println("Optimal Hitting Time: ",tau_0)
println("Rank of the solution matrix X: ",rank(solution))
