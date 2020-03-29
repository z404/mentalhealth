import functools as f
Q=input
W=int
E=range
R=print
T=sum
Y=map
a=W(Q())
while(a>0):
    b,a=Q().split(),a-1
    m,n=W(b[0]),W(b[1])
    if n>0and n<101and m>0and m<101:
        A,B,v=[],[],0
        for i in E(m):A.append([i for i in Y(W,Q())])
        for i in E(m):B.append([i for i in Y(W,Q())])
        c,l=T(Y(T,A)),T(Y(T,B))
        if f.reduce(lambda i,j:i and j,Y(lambda m,k:m==k,A,B),1):p=1
        else:p=0
        if(c==l and p!=1):
            for i in E(m):
                for j in E(n):
                    if(A[i][j]!=B[i][j]):v+=1
            R(v//2)
        elif(p==1):R("0")
        else:R("-1")
