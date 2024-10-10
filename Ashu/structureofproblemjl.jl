using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS

N = 2
λ = 1
ni = 4 * (N + 1)^2
np = 5 * N + 1
ns  = ni - 12
nt = np + ns + 1


S = Dict()
T = []
TG = []
TB = []
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
                S[i] = (ua, ub, sa, sb)
            end
        end
    end
end

Tc = [i for i in 0:ni-1 if i ∉ T]

sort!(T);
sort!(TG);
sort!(TB);


f⁺ = collect([[N * (j-1) + i for j in 1:2] for i in 1:N]);
f⁺=reduce(hcat,f⁺)
f⁻ = collect([[N * (j-1) + i + (2 * N) for j in 1:2] for i in 1:N]);
f⁻=reduce(hcat,f⁻)
f⁺_flattened = collect(Iterators.flatten(f⁺'))
f⁻_flattened = collect(Iterators.flatten(f⁻'))
g = [4 * N  + i for i in 1:N];
koff = 5 * N + 1
P = collect(vcat(f⁺_flattened, f⁻_flattened, g, [koff]))


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

    if j == i + (N + 1) && ub == ub_ && sa == sa_ && sb == sb_
        flag = 1
        tempk = N * sa + ua + 1
    elseif j == i - (N + 1) && ub == ub_ && sa == sa_ && sb == sb_
        flag = 1
        tempk = sa * N + ua + 2 * N 
    elseif j == i + 1 && ua == ua_ && sa == sa_ && sb == sb_
        flag = 1
        tempk = N * sb + ub + 1
    elseif j == i - 1 && ua == ua_ && sa == sa_ && sb == sb_
        flag = 1
        tempk = sb * N + ub + 2 * N 
    elseif j == i + 2 * (N + 1)^2 && ua == ua_ && ub == ub_ && ub != 0 && sb == sb_
        flag = 1
        tempk = 4 * N + ub 
    elseif j == i - 2 * (N + 1)^2 && ua == ua_ && ub == ub_ && sb == sb_
        flag = 1
        tempk = 5 * N + 1
    elseif j == i + (N + 1)^2 && ua == ua_ && ub == ub_ && ua != 0 && sa == sa_
        flag = 1
        tempk = 4 * N + ua 
    elseif j == i - (N + 1)^2 && ua == ua_ && ub == ub_ && sa == sa_
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

P_=reshape(P, np, 1)
Q = reduce((x, y) -> x + y, [P_[k] * M[:, :, k] for k in 1:np])

R = zeros(ns, ni)
for i in 1:ns
    R[i,Tc[i]+1] = 1
end
Qtilde = R * Q * R'

Mtilde = zeros(ns, ns, np)
for i in 1:ns
    for j in 1:ns
        for k in 1:np
            Mtilde[i, j, k] = sum(R[i, a] * R[j, b] * M[a, b, k] for a in 1:ni, b in 1:ni)
        end
    end
end

eis = Matrix{Float64}(I, ns, ns)
Ai = zeros(ns, np, ns)
for k in 1:ns
    ei=eis[k,:]
    for i in 1:ns
        for j in 1:np
            Ai[i,j,k]=ei'* Mtilde[:,i,j]   
        end 
    end
end

alpha=zeros(np,ns)
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

Di=zeros(nt,nt,ns)
for i in 1:ns
    Di[np+1:np+ns,1:np,i]=Ai[:,:,i]
    Di[1:np,nt,i]=λ*alpha[:,i]'
    Di[nt,nt,i]=1
end

Ei=zeros(nt,nt,nt)
for i in 1:nt
    Ei[i,nt,i]=1
end
S0=Ei[:,:,np+1]

Cis=zeros(nt,nt,np)
for i in 1:np
    Cis[i,i,i]=1
end

