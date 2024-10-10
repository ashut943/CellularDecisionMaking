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

print("Sizes")
print('N',N)
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

print("Paramters")
print('f+',f_plus)
print('f-',f_min)
print('g',g)
print('k_off',koff)
print('P',P)

print("======================")

#~~~~~~~~~Rate Matrix

# Q=[[0 for j in range(n_I)]for i in range(n_I)]
# Q2=[[0 for j in range(n_I)]for i in range(n_I)]
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
        
        # if(j==i+(N+1) and ub==ub_ and sa==sa_ and sb==sb_ ):
        #     flag=P[(N)*sa+ua]
        # elif(j==i-(N+1) and ub==ub_ and sa==sa_ and sb==sb_ ):
        #     flag=P[sa*N+ua+2*N-1]
        # elif(j==i+1 and ua==ua_ and sa==sa_ and sb==sb_ ):
        #     flag=P[(N)*sb+ub]
        # elif(j==i-1 and ua==ua_ and sa==sa_ and sb==sb_ ):
        #     flag=P[sb*N+ub+2*N-1]
        # elif(j==i+2*(N+1)**2 and ua==ua_ and ub==ub_ and ub!=0 and sb==sb_ ):
        #     flag=P[4*N+ub-1]
        # elif(j==i-2*(N+1)**2 and ua==ua_ and ub==ub_ and sb==sb_ ):
        #     flag=P[5*N]   
        # elif(j==i+(N+1)**2 and ua==ua_ and ub==ub_ and ua!=0  and sa==sa_ ):
        #     flag=P[4*N+ua-1]
        # elif(j==i-(N+1)**2 and ua==ua_ and ub==ub_ and sa==sa_ ):
        #     flag=P[5*N]   
        ##To check definitions
        # Q2[i][j]=flag

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

        # flag=0
        # if(ua_==ua+1 and ub==ub_ and sa==sa_ and sb==sb_):
        #     flag=f_plus[ua][sa]
        # elif(ua_==ua-1 and ub==ub_ and sa==sa_ and sb==sb_):
        #     flag=f_min[ua-1][sa]
        # elif(ub_==ub+1 and ua==ua_ and sa==sa_ and sb==sb_):
        #     flag=f_plus[ub][sb]
        # elif(ub_==ub-1 and ua==ua_ and sa==sa_ and sb==sb_):
        #     flag=f_min[ub-1][sb]
        # elif(ub_==ub and ua==ua_ and ub!=0 and((sa==0 and sa_==1 and sb==sb_))):
        #     flag=g[ub-1]
        # elif(ub_==ub and ua==ua_ and ua!=0 and ((sb==0 and sb_==1 and sa==sa_))):
        #     flag=g[ua-1]
        # elif(ub_==ub and ua==ua_ and ((sa==1 and sa_==0 and sb==sb_) or (sb==1 and sb_==0 and sa==sa_))):
        #     flag=koff
        # Q[i][j]=flag
        # #To check definitions

# Q=np.array(Q)
# Q2=np.array(Q2)
M=np.array(M)

# Q_i = [sum(row) for row in Q]
# Q2_i=[sum(row) for row in Q2]

for u in Ivals:
        i=Ikeyer[u]
        # Q[i][i]=-Q_i[i]
        # Q2[i][i]=-Q2_i[i]


for k in range(n_p):
    M_i=[sum(row) for row in M[:,:,k]]
    for u in Ivals:
        i=Ikeyer[u]
        M[i][i][k]=-M_i[i]


# print('Q=',Q)
# print("======================")
# print('Q=',Q2)
# print("======================")
# Q_=0
# for k in range(n_p):
#     Q_+=P[k]*np.array(M[:,:,k])
# Q_=np.array(Q_)
# # print('Q=',Q_)
# # print("======================")
# # for k in range(n_p):
# #     print('k=',k,',Mk=',M[k])
# #     print("~~~~~~~~~~~~~~")
# print("======================")
Q__=0
P_=np.array(P).reshape((n_p,1)) 
Q__=np.tensordot(P_,M,axes=((0),(2)))
Q__=np.array(Q__[0])
print('Q=',Q__)
print("======================")
# print(np.max(np.abs(np.array(Q)-np.array(Q2))))
# print(np.max(np.abs(np.array(Q)-np.array(Q_))))
# print(np.max(np.abs(np.array(Q)-np.array(Q__))))
# print(np.max(np.abs(np.array(Q__)-np.array(Q_))))
# # print_matrix_latex(Q)
# print("======================")

#~~~~~~~~~Define Further Matrices for the general non-liner problem

Q=Q__
R=[[0 for j in range(n_I)]for i in range(n_s)]
Tc=[i for i in range(n_I) if i not in T]
i=0
for j in Tc:
    R[i][j]=1
    i+=1

R=np.array(R)
# print(R.shape)
# print('R=',R)
# print("======================")
#Qtilde=RQR^T
Qtilde=R@Q@(R.T)
# #Mtilde_ijk=RiaRjbMabk
# #but M=M[k][i][j]=M_ijk
#shape of M is n_I*n_I*np
#shape of R is ns*nI
#so Mtilde is ns*ns*np
Mtilde=np.einsum('ia,jb,abk->ijk', R, R, M)
# print(Mtilde.shape)
# #Qtilde=Mtilde.P
#so qtilde is ns*ns
Qtilde_=np.array(np.tensordot(P_,Mtilde,axes=((0),(2))))[0]
# print(Qtilde_.shape)
# print(np.max(np.abs(Qtilde-Qtilde_)))
# print("======================")
eis=[[0 for j in range(n_s)]for i in range(n_s)]
eis=np.array(eis)
for i in range(n_s):
    eis[i][i]=1
# print(eis)
# print(M.shape)
#each Ai is ns times np
#each alphai is n_p vector and total n_s such vectors
Ai=[np.tensordot(np.array(eis[i]).reshape((n_s,1)),Mtilde,axes=((0),(0)))[0] for i in range(n_s)]
Ai=np.array(Ai)
# print(Ai[0].shape)
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
# print("======================")
A_tilde_0 = Ai[0].copy()
A_tilde_0[0, :] = 0
# print(A_tilde_0.shape)
print("======================")

#~~~~~~~~~Matrices for the our optimisation problem - first formulation 

D_i=[np.zeros((n_t, n_t)) for i in range(n_s)]
D_i=np.array(D_i)
for i in range(n_s):
    D_i[i,n_p:n_p+n_s, :n_p] = Ai[i]              
    D_i[i,0:n_p, -1] = lmbda * (alphai[i]).reshape(1,n_p)  
    D_i[i,-1, -1] = 1 

S_0=np.zeros((n_t, n_t))
S_0[n_p:n_p+n_s, :n_p] = A_tilde_0            
S_0[0:n_p, -1] = lmbda * (alphai[0]).reshape(1,n_p)  
S_0[-1, -1] = 1 

# print(D_i.shape)
# print(S_0.shape)
print("======================")
#~~~~~~~~~Our optimisation problem - first formulation 
tautilde=np.array([0 for i in range(n_s)])
Z=list(P)+list(tautilde)+[1]
Z=np.array(Z)
Z_=Z.reshape(n_t,1)
# print(Z)
# print(Z.shape)
# for i in range(n_s):
#     print(Z.T@D_i[i]@Z)
# print(Z.T@S_0@Z)
# print("======================")
# print("======================")
Ci=[np.zeros((n_t,n_t)) for i in range(n_p)]
Ci=np.array(Ci)
for i in range(n_p):
    Ci[i,i,i]=1
# for i in range(n_p):
#     print(Z.T@Ci[i]@Z)

print("======================")
#~~~~~~~~~Matrices for optimisation problem - second (symmetricised) formulation 

Di=np.copy(D_i)
for i in range(n_s):
    Di[i]=(D_i[i]+D_i[i].T)/2
S0=(S_0+S_0.T)/2
# for i in range(n_s):
#     print(Z.T@Di[i]@Z)
# print(Z.T@S0@Z)




