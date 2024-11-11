using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs

N = 3
λ = 0.15
ni = 4 * (N + 1)^2
np = 5 * N + 1
ns  = ni - 12
nt = np + ns + 1
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

function hitting_time(Q,targetstates_good,targetstates_bad,startstates,λ)
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
    T = A \ b
    for i in 1:n
        if T[i]==0 && i ∉ targetstates
            T[i]=Inf 
        end
    end
    return [T[start_state] for start_state ∈ startstates]
end

function is_irreducible(Q::Matrix{Float64})
    n = size(Q, 1)
    G = SimpleDiGraph(n) 
    for i in 1:n
        for j in 1:n
            if i != j && Q[i, j] > 0
                add_edge!(G, i, j)
            end
        end
    end
    return is_strongly_connected(G)
end

function Q_maker(P,N::Int64,λ::Float64,S,Skeyer)
    ni = 4 * (N + 1)^2
    np = 5 * N + 1
    ns  = ni - 12
    nt = np + ns + 1
    # Q=@expression(model, zeros(AffExpr, ni, ni)) 
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
        elseif ua == ua_ && ub == ub_ && ub != 0 && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub 
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 1
        elseif ua == ua_ && ub == ub_ && ua != 0 && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua 
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 1
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

function M_maker(N::Int64,λ::Float64,S,Skeyer)
    ni = 4 * (N + 1)^2
    np = 5 * N + 1
    ns  = ni - 12
    nt = np + ns + 1
    M=zeros(Int, ni, ni, np);
    for (u, u_) in Iterators.product(values(S), values(S))
        # Skip if u and u_ are the same
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
        elseif ua == ua_ && ub == ub_ && ub != 0 && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub 
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 1
        elseif ua == ua_ && ub == ub_ && ua != 0 && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua 
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 1
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
    M=M_maker(N::Int64,λ::Float64,S,Skeyer)
    Q = reduce((x, y) -> x + y, [P[k] * M[:, :, k] for k in 1:np])
    return Q
end

function Q_maker_original_mod(P,N::Int64,λ::Float64, model,S,Skeyer)
    ni = 4 * (N + 1)^2
    np = 5 * N + 1
    ns  = ni - 12
    nt = np + ns + 1
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
        elseif ua == ua_ && ub == ub_ && ub != 0 && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub 
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 1
        elseif ua == ua_ && ub == ub_ && ua != 0 && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua 
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 1
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

function M_maker_mod(N::Int64,λ::Float64, model,S,Skeyer)
    ni = 4 * (N + 1)^2
    np = 5 * N + 1
    ns  = ni - 12
    nt = np + ns + 1
    M=@expression(model, zeros(AffExpr, ni, ni, np)) 
    for (u, u_) in Iterators.product(values(S), values(S))
        # Skip if u and u_ are the same
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
        elseif ua == ua_ && ub == ub_ && ub != 0 && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub 
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 1
        elseif ua == ua_ && ub == ub_ && ua != 0 && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua 
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 1
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

function Q_maker_using_M_mod(P,N::Int64,λ::Float64,model,S,Skeyer)
    M=M_maker_mod(N,λ,model,S,Skeyer)
    Q = reduce((x, y) -> x + y, [P[k] * M[:, :, k] for k in 1:np])
    return Q
end

function Q_maker_tilde_mod(P,N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    ni = 4 * (N + 1)^2
    np = 5 * N + 1
    ns  = ni - 12
    nt = np + ns + 1

    #Get Q
    Q=Q_maker_original_mod(P,N,λ, model,S,Skeyer)

    #Get Restricter R

    R = zeros(ns, ni)
    for i in 1:ns
        R[i,Tc[i]+1] = 1
    end

    Qtilde = R * Q * R'
    return Qtilde
end

function M_maker_tilde_mod(N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    ni = 4 * (N + 1)^2
    np = 5 * N + 1
    ns  = ni - 12
    nt = np + ns + 1

    #Get M
    M=M_maker_mod(N,λ,model,S,Skeyer)

    #Get Restricter R

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

function Q_maker_tilde_2_mod(P,N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    np = 5 * N + 1
    Mtilde=M_maker_tilde_mod(N,λ, model,S,Skeyer,T,TG,TB,Tc)
    return reduce((x, y) -> x + y, [P[k] * Mtilde[:, :, k] for k in 1:np])
end

function A_maker_mod(N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    ni = 4 * (N + 1)^2
    np = 5 * N + 1
    ns  = ni - 12
    nt = np + ns + 1
    Mtilde=M_maker_tilde_mod(N,λ, model,S,Skeyer,T,TG,TB,Tc)
    Ai = @expression(model, zeros(AffExpr, ns, np, ns)) 
    for i in 1:ns
        Ai[:,:,i]= Mtilde[i,:,:]   
    end
    return Ai
end

function alpha_maker_mod(N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    ni = 4 * (N + 1)^2
    np = 5 * N + 1
    ns  = ni - 12
    nt = np + ns + 1
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

function hitting_time_mod(Q,targetstates_good,targetstates_bad,startstates,λ,model)
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

model = Model(Ipopt.Optimizer)
f⁺=rand(N, 2)
f⁻=rand(N, 2)
f⁺_flattened = collect(Iterators.flatten(f⁺'))
f⁻_flattened = collect(Iterators.flatten(f⁻'))
g=rand(N)
koff=rand(Uniform(0,1))
P = collect(vcat(f⁺_flattened, f⁻_flattened, g, [koff]))
targetstates_good=[target_state+1 for target_state ∈ TG];
targetstates_bad=[target_state+1 for target_state ∈ TB];
startstates=[start_state+1 for start_state ∈ Tc];
allstates=[startstates;targetstates_good; targetstates_bad]

model = Model(Ipopt.Optimizer)
f⁺=rand(N, 2)
f⁻=rand(N, 2)
f⁺_flattened = collect(Iterators.flatten(f⁺'))
f⁻_flattened = collect(Iterators.flatten(f⁻'))
g=rand(N)
koff=rand(Uniform(0,1))
P = collect(vcat(f⁺_flattened, f⁻_flattened, g, [koff]))
targetstates_good=[target_state+1 for target_state ∈ TG];
targetstates_bad=[target_state+1 for target_state ∈ TB];
startstates=[start_state+1 for start_state ∈ Tc];
allstates=[startstates;targetstates_good; targetstates_bad]

# Q=Q_maker_original_mod(P,N,λ,model,S,Skeyer)
Q_exact=Q_maker(P,N,λ,S,Skeyer)
Q2=Q_maker_using_M_mod(P,N,λ,model,S,Skeyer)

# println(maximum(abs.(Qtilde2-Qtilde)))

τ_actual=hitting_time(Q_exact,targetstates_good,targetstates_bad,allstates,λ)
τ_now=hitting_time_mod(Q2,targetstates_good,targetstates_bad,allstates,λ,model)
model = Model(Ipopt.Optimizer)
# set_optimizer_attribute(model, "print_level", 0)
targetstates_good=[target_state+1 for target_state ∈ TG];
targetstates_bad=[target_state+1 for target_state ∈ TB];
targetstates=[targetstates_good;targetstates_bad]
startstates=[start_state+1 for start_state ∈ Tc];
allstates=[startstates;targetstates_good; targetstates_bad]

# Q=Q_maker_original_mod(P,N,λ,model,S,Skeyer)
# Q_exact=Q_maker(P,N,λ,S,Skeyer)

# alpha=alpha_maker_mod(N,λ, model,S,Skeyer,T,TG,TB,Tc)
# Ai=A_maker_mod(N,λ, model,S,Skeyer,T,TG,TB,Tc)

# Qtilde=Q_maker_tilde_mod(P,N,λ,model,S,Skeyer,T,TG,TB,Tc)
# Qtilde2=Q_maker_tilde_2_mod(P,N,λ,model,S,Skeyer,T,TG,TB,Tc)
# Mtilde=M_maker_tilde_mod(N,λ, model,S,Skeyer,T,TG,TB,Tc)

# n=size(Q,1)
# @variable(model, τ[1:ns])
# for i=1:ns
#     @constraint(model, 1 + τ'*Ai[:,:,i]*P + λ*alpha[:,i]'*P == 0)
# end
# optimize!(model)
# τ_now=value.(τ)
# println("τ_0: ",τ_now[1])
# println("τ_0,exact: ",τ_actual[1])

model = Model(Ipopt.Optimizer)
# set_optimizer_attribute(model, "print_level", 0)
targetstates_good=[target_state+1 for target_state ∈ TG];
targetstates_bad=[target_state+1 for target_state ∈ TB];
targetstates=[targetstates_good;targetstates_bad]
startstates=[start_state+1 for start_state ∈ Tc];
allstates=[startstates;targetstates_good; targetstates_bad]
alpha=alpha_maker_mod(N,λ, model,S,Skeyer,T,TG,TB,Tc)
Ai=A_maker_mod(N,λ, model,S,Skeyer,T,TG,TB,Tc)
@variable(model, 1>=P_[1:np]>=0) 
@variable(model, τ[1:ns])
@objective(model, Min, τ[1])
for i=1:ns
    @constraint(model, 1 + τ'*Ai[:,:,i]*P_ + λ*alpha[:,i]'*P_ == 0)
end
optimize!(model)
P_opt=value.(P_);
tau_opt=value.(τ);
println(tau_opt[1])