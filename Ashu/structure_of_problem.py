import numpy as np
import sys

def print_matrix_latex(matrix):
    max_width = max(len(str(item)) for row in matrix for item in row)
    for row in matrix:
        print(" & ".join(f"{str('$')+str(item)+str('$'):>{max_width}}" for item in row))
        print('\\\\')


np.set_printoptions(precision=3, suppress=True,threshold=sys.maxsize, linewidth=np.inf)

#----Sizes and N
N=1
lmbda=1
n_I=4*(N+1)**2
n_p=5*N+1
n_s=n_I-12
n_P=5*N-1
n_t=n_p+n_s+1

print("======================")

print("Sizes and cost")
print('N',N)
print('lambda',lmbda)
print('nI',n_I)
print('np',n_p)
print('ns',n_s)
print('nP',n_P)
print('nt',n_t)


print("======================")

#~~~~~~~~~States 

I={}
T=[]
TG=[]
TB=[]
for ua in range(N+1):
    for ub in range(N+1):
        for sa in range(2):
            for sb in range(2):
                index=(2*sa+sb)*(N+1)**2+(N+1)*ua+ub
                I[index]=(ua,ub,sa,sb)

I=(dict(sorted(I.items())))
Ivals=list(I.values())
Ikeyer = dict((v,k) for k,v in I.items())
print("State Definitions")
for key, value in I.items():
    print("All states")
    print(f"{key}: {value}")

print("~~~~~~~~~~~~~~~~~~~~~")


#now try defining terminal states and TG TB
for ua in range(N+1):
    for ub in range(N+1):
        for sa in range(2):
            for sb in range(2):
                i=(2*sa+sb)*(N+1)**2+(N+1)*ua+ub
                if(ua==N and ub==0):
                    TG.append(i)
                    T.append(i)
                if(ua==0 and ub==N):
                    TG.append(i)
                    T.append(i)
                if(ua==N and ub==N):
                    TB.append(i)
                    T.append(i)
                I[index]=(ua,ub,sa,sb)
T.sort()
TG.sort()
TB.sort()
print("Terminal States")
print('T',T)
print('TG',TG)
print('TB',TB)

print("======================")

#~~~~~~~~~Parameters

f_plus=[[1+N*j+i for j in range(2)] for i in range (N)]
f_min=[[N*j+i+(2*N)+1 for j in range(2)] for i in range (N)]

f_plus_flattened = [f_plus[i][j] for j in range(2) for i in range(N)]
f_min_flattened = [f_min[i][j] for j in range(2) for i in range(N)]
g=[4*N +1+i for i in range(N)]
koff=5*N+1
P = f_plus_flattened + f_min_flattened + g + [koff]
P=np.array(P)

print("Parameters")
print('f+',f_plus)
print('f-',f_min)
print('g',g)
print('k_off',koff)
print('P',P)

print("======================")

#~~~~~~~~~Rate Matrix

M=[[[0 for k in range(n_p)]for j in range(n_I)] for i in range(n_I)]

alpha=[0 for i in range(n_I)]

for u in Ivals:
    for u_ in Ivals:
        if(u==u_):
            continue
        i=Ikeyer[u]
        j=Ikeyer[u_]
        flag=0
        ua=u[0]
        ub=u[1]
        sa=u[2]
        sb=u[3]
        ua_=u_[0]
        ub_=u_[1]
        sa_=u_[2]
        sb_=u_[3]

        flag=0
        tempk=0
        if(j==i+(N+1) and ub==ub_ and sa==sa_ and sb==sb_ ):
            flag=1
            tempk=(N)*sa+ua
        elif(j==i-(N+1) and ub==ub_ and sa==sa_ and sb==sb_ ):
            flag=1
            tempk=sa*N+ua+2*N-1
        elif(j==i+1 and ua==ua_ and sa==sa_ and sb==sb_ ):
            flag=1
            tempk=(N)*sb+ub
        elif(j==i-1 and ua==ua_ and sa==sa_ and sb==sb_ ):
            flag=1
            tempk=sb*N+ub+2*N-1
        elif(j==i+2*(N+1)**2 and ua==ua_ and ub==ub_ and ub!=0 and sb==sb_ ):
            flag=1
            tempk=4*N+ub-1
        elif(j==i-2*(N+1)**2 and ua==ua_ and ub==ub_ and sb==sb_ ):
            flag=1 
            tempk=5*N
        elif(j==i+(N+1)**2 and ua==ua_ and ub==ub_ and ua!=0  and sa==sa_ ):
            flag=1
            tempk=4*N+ua-1
        elif(j==i-(N+1)**2 and ua==ua_ and ub==ub_ and sa==sa_ ):
            flag=1   
            tempk=5*N
        M[i][j][tempk]=flag

M=np.array(M)

for k in range(n_p):
    M_i=[sum(row) for row in M[:,:,k]]
    for u in Ivals:
        i=Ikeyer[u]
        M[i][i][k]=-M_i[i]

Q__=0
P_=np.array(P).reshape((n_p,1)) 
Q__=np.tensordot(P_,M,axes=((0),(2)))
Q__=np.array(Q__[0])
print(Q__)
print("======================")

#~~~~~~~~~Define Further Matrices for the general non-liner problem

Q=Q__
R=[[0 for j in range(n_I)]for i in range(n_s)]
Tc=[i for i in range(n_I) if i not in T]
i=0
for j in Tc:
    R[i][j]=1
    i+=1

R=np.array(R)
Qtilde=R@Q@(R.T)
Mtilde=np.einsum('ia,jb,abk->ijk', R, R, M)
Qtilde_=np.array(np.tensordot(P_,Mtilde,axes=((0),(2))))[0]
eis=[[0 for j in range(n_s)]for i in range(n_s)]
eis=np.array(eis)
for i in range(n_s):
    eis[i][i]=1
Ai=[np.tensordot(np.array(eis[i]).reshape((n_s,1)),Mtilde,axes=((0),(0)))[0] for i in range(n_s)]
Ai=np.array(Ai)
alphai=[[0 for j in range(len(TB))] for i in range((n_s))]
Tc=[i for i in range(n_I) if i not in T]
for ind in range(n_s):
    i=Tc[ind]
    alphai_temp=[0 for k in range(n_p)]
    for k in range(n_p):
        for j in TB:
            alphai_temp[k]+=M[i,j,k]
    alphai[ind]=alphai_temp
alphai=np.array(alphai)
print("======================")

#~~~~~~~~~Matrices for the our optimisation problem - first formulation 

D_i=[np.zeros((n_t, n_t)) for i in range(n_s)]
D_i=np.array(D_i)
for i in range(n_s):
    D_i[i,n_p:n_p+n_s, :n_p] = Ai[i]              
    D_i[i,0:n_p, -1] = lmbda * (alphai[i]).reshape(1,n_p)  
    D_i[i,-1, -1] = 1 

E_i=np.array([np.zeros((n_t, n_t)) for i in range(n_t)])

for i in range(n_t):
    E_i[i,i,-1]=1

S_0=E_i[n_p,:,:]

print("======================")
#~~~~~~~~~Matrices for optimisation problem - symmetricised formulation 

Di=np.copy(D_i)
for i in range(n_s):
    Di[i]=(D_i[i,:,:]+D_i[i,:,:].T)/2

S0=(S_0+S_0.T)/2

Ei=np.copy(E_i)
for i in range(n_t):
    Ei[i]=(E_i[i,:,:]+E_i[i,:,:].T)/2


print("======================")
#~~~~~~~~~Our optimisation problem - first symmetricised formulation 
tautilde=np.array([0 for i in range(n_s)])
Z=list(P)+list(tautilde)+[1]
Z=np.array(Z)
Z_=Z.reshape(n_t,1)

Ci=[np.zeros((n_t,n_t)) for i in range(n_p)]
Ci=np.array(Ci)
for i in range(n_p):
    Ci[i,i,i]=1

#Evaluate stuff by
# Z.T@S0@Z
# Z.T@Di[i]@Z, i in range(n_s)
# Z.T@Ci[i]@Z, i in range(n_p)
# Z.T@Ei[i]@Z, i in range(n_t)

print("======================")




