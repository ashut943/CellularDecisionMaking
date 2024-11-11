using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO
include("utils.jl")
include("two_cell_functions_ctmc.jl")

N=5
λ=5.0
upper_bound=11.0

# model = Model(SCS.Optimizer)
model = Model(COSMO.Optimizer)

#We note that we have added a few more constraints here!


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

U=zeros(nt)
L=zeros(nt)

for i in 1:nt-1
    if(i<=np)
        U[i]=1.0
    else
        U[i]=upper_bound
    
    end
end
U[nt]=1

@variable(model, X[1:nt, 1:nt], Symmetric)#Sym
@variable(model, L[i] <= Z[i=1:nt] <= U[i])

@objective(model, Min, Z[np+1])

@constraint(model, X in PSDCone())#PSD
@constraint(model, X .>= 0) #DNN
for i=1:ns
    @constraint(model, tr(X * Di[:,:,i]) == 0.0)
end
for i=1:nt-1
    for j=1:np
        @constraint(model, X[i,j] <= X[i,nt])
    end
end
@constraint(model, X[nt,nt] == 1)
@constraint(model, Z[nt] == 1)
@constraint(model, [i=1:nt, j=1:i-1], X[i,j] == X[j,i])
@constraint(model, [i=1:nt, j=1:nt; i != j], X[i,j] >= L[i]*Z[j] + L[j]*Z[i] - L[i]*L[j])
@constraint(model, [i=1:nt, j=1:nt; i != j], X[i,j] >= U[i]*Z[j] + U[j]*Z[i] - U[i]*U[j])
@constraint(model, [i=1:nt, j=1:nt; i != j], X[i,j] <= L[i]*Z[j] + U[j]*Z[i] - L[i]*U[j])
@constraint(model, [i=1:nt, j=1:nt; i != j], X[i,j] <= U[i]*Z[j] + L[j]*Z[i] - U[i]*L[j])



println("Ok problem set up done!")
optimize!(model)
solution = value.(X)
tau_0=objective_value(model)
println("Optimal Hitting Time: ",tau_0)
println("Rank of the solution matrix X: ",rank(solution))
