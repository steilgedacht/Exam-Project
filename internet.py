import numpy
import sympy

# source: https://mathsfromnothing.au/sequential-quadratic-programing-methods/?i=1

def sqp(f,x,ce=[],ci=[],df=[],ddf=[],dce=[],ddce=[],dci=[],ddci=[]):
    xtol=1e-8
    ytol=1e-8
    Delta=5
    DeltaM=numpy.inf
    t1=.25
    t2=2
    eta1=.2
    eta2=.25
    eta3=.75
    maxit=1000
    mu=1e-2
    delta=1
    n=len(x)
    if not callable(ce):
        ce=lambda x:[]
        dce=lambda x:[]
        ddec=lambda x:[]
    elif not callable(dce):
        ne=len(ce(x))
        dce=symconstraint(ce,n,ne)
        ddce=[]
        for k in range(ne):
            sddc=symJacobianHessian(dce[:,k],n)
            nothing,nothing1,sddc=symtolambda(lambda x:0,lambda x:0,n,sddc)
            ddce.append(sddc)
        nothing,ce,dce=symtolambda(lambda x:0,ce,n,dce)
    ne=len(ce(x))
    if ne!=0 and ne!=len(ddce):
        ddce=[]
        for k in range(ni):
            sddc=symJacobianHessian(lambda x:dce(x)[:,k],n)
            nothing,nothing1,sddc=symtolambda(lambda x:0,lambda x:0,n,sddc)
            ddce=ddce.append(sddc)
    if not callable(ci):
        ci=lambda x:[]
        dci=lambda x:[]
        ddci=lambda x:[]
    elif not callable(dci):
        ni=len(ci(x))
        dci=symconstraint(ci,n,ni)
        ddci=[]
        for k in range(ni):
            sddc=symJacobianHessian(dci[:,k],n)
            nothing,nothing1,sddc=symtolambda(lambda x:0,lambda x:0,n,sddc)
            ddci.append(sddc)
        nothing,ci,dci=symtolambda(lambda x:0,ci,n,dci)
    ni=len(ci(x))
    if ni!=0 and ni!=len(ddci):
        ni=len(ci(x))
        ddci=[]
        for k in range(ni):
            sddc=symJacobianHessian(lambda x:numpy.array(dci(x))[:,k],n)
            nothing,nothing1,sddc=symtolambda(lambda x:0,lambda x:0,n,sddc)
            ddci.append(sddc)
    if not callable(df):
        df=symGradient(f,n)
        ddf=symJacobianHessian(df,n)
    elif not callable(ddf):
        ddf=symJacobianHessian(df,n)
    f,df,ddf=symtolambda(f,df,n,ddf)
    ne=len(ce(x))
    ni=len(ci(x))

    lambd=numpy.zeros(ne+ni)
    cxnormold=numpy.inf
    fx,g,B=alagrangian(x,lambd,mu,ne,ni,f,df,ddf,ce,dce,ddce,ci,dci,ddci)
    for k in range(maxit):
        p=gradientprojection(numpy.zeros(n),g,B,-Delta*numpy.ones(n),Delta*numpy.ones(n))
        fp=taylorexpansionquadratic(p,fx,g,B)
        fxp,gp,Bp=alagrangian(x+p,lambd,mu,ne,ni,f,df,ddf,ce,dce,ddce,ci,dci,ddci)
        fpfx=fp-fx
        if numpy.abs(fpfx)>0:
            compare=(fxp-fx)/fpfx
        else:
            compare=numpy.maximum(eta1+numpy.spacing(eta1),eta2)
        if compare<eta2:
            Delta=t1*Delta
        elif compare>eta3 and numpy.any(numpy.abs(numpy.abs(p)-Delta)<1e-8):
            Delta=numpy.minimum(t2*Delta,DeltaM)
        if compare>eta1:
            x=x+p
            if ne>0:
                dcexp=dce(x)
            else:
                dcexp=numpy.zeros((n,0))
            if ni>0:
                dcixp=dci(x)
            else:
                dcixp=numpy.zeros((n,0))
            A=numpy.transpose(numpy.concatenate((dcexp,dcixp),1))
            lambd=numpy.linalg.solve(numpy.matmul(A,numpy.transpose(A)),numpy.matmul(A,df(x+p)))
            ccii=ci(x)-mu*lambd[ne:ne+ni]
            ccii[ccii<0]=0
            cx=numpy.concatenate((ce(x),ci(x)-ccii))
            cxnorm=numpy.dot(cx,cx)**.5
            fx,g,B=alagrangian(x,lambd,mu,ne,ni,f,df,ddf,ce,dce,ddce,ci,dci,ddci)
            if cxnorm/cxnormold>.8 or cxnorm>numpy.dot(g,g)**.5:
                mu=numpy.maximum(.1*mu,1e-10)
                fx,g,B=alagrangian(x,lambd,mu,ne,ni,f,df,ddf,ce,dce,ddce,ci,dci,ddci)
            cxnormold=cxnorm
        if numpy.dot(g,g)**.5<ytol and numpy.dot(p,p)**.5<xtol and cxnorm<ytol:
            return x,lambd,fx,g
        if numpy.max(numpy.abs(p))<1e1*numpy.spacing(numpy.max(numpy.abs(x))) and ~numpy.all(p==0):
            print("Ended becaues change is to small")
            return x,lambd,fx,g
    if k==maxit:
        print("Exceeded maximum number of iterations")
        return x,lambd,fx,g
    if k==maxit:
        print('Maximum iterations reached and no solution found.')
    return

def taylorexpansionquadratic(p,fx,g,B):
    return fx+numpy.dot(p,g)+.5*numpy.dot(p,numpy.matmul(B,p))

def alagrangian(x,lambd,mu,ne,ni,f,df,ddf,ce,dce,ddce,ci,dci,ddci):
    fx=f(x)
    dfx=numpy.array(df(x))
    ddfx=numpy.array(ddf(x))
    cex=numpy.array(ce(x))
    dcex=numpy.array(dce(x))
    cix=numpy.array(ci(x))
    dcix=numpy.array(dci(x))
    elambda=numpy.array(lambd[0:ne])
    ilambda=numpy.array(lambd[ne:ne+ni])
    s=cix-mu*ilambda
    s[s<0]=0
    constrained=s==0
    L=fx\
    -numpy.sum(cex*elambda)+numpy.sum(cex**2)/(2*mu)\
    -numpy.sum((cix-s)*ilambda)+numpy.sum((cix-s)**2)/(2*mu)

    dL=dfx
    if ne>0:
        dL=dL-numpy.matmul(dcex,elambda)\
        +numpy.matmul(dcex,cex)/mu
    if ni>0:
        dL=dL-numpy.matmul(constrained*dcix,ilambda)\
        +numpy.matmul(constrained*dcix,cix)/mu

    ddL=ddfx
    for j in range(ne):
        ddL=ddL-elambda[j]*numpy.array(ddce[j](x))
        ddL=ddL+(numpy.array(ddce[j](x))*ce(x)[j])/mu
    if ne>0:
        ddL=ddL+numpy.matmul((constrained*dce(x)),numpy.transpose(dce(x)))/mu

    for j in range(ni):
        ddL=ddL+constrained[j]*(-ilambda[j]*numpy.array(ddci[j](x)))
        ddL=ddL+constrained[j]*numpy.array(ddci[j](x))*cix[j]/mu
    if ni>0:
        ddL=ddL+numpy.matmul((constrained*dci(x)),numpy.transpose(dci(x)))/mu
    return L,dL,ddL

def gradientprojection(x,d,G,l,u):
    tol=1e-14*numpy.max(numpy.max(numpy.abs(G)))
    n=len(x)
    x=numpy.array(x)
    d=numpy.array(d)
    G=numpy.array(G)
    l=numpy.array(l)
    u=numpy.array(u)
    g=numpy.matmul(G,x)+d
    while True:
        tb=numpy.inf*numpy.ones(n)
        tb[(g<0)&(u<numpy.inf)]=((x-u)/g)[(g<0)&(u<numpy.inf)]
        tb[(g>0)&(l>-numpy.inf)]=((x-l)/g)[(g>0)&(l>-numpy.inf)]
        tbi=numpy.argsort(tb)
        tbs=tb[tbi]
        xt=lambda t:x-numpy.minimum(tb,t)*g
        xm=numpy.array(x)
        gm=numpy.array(g)
        tminf=0
        for k in range(n):
            gGg=numpy.dot(gm,numpy.matmul(G,gm))
            mc=numpy.dot(gm,d)+numpy.dot(xm,numpy.matmul(G,gm))
            tmid=mc/gGg
            if tmid>0 and tmid<tbs[k] and gGg>0:
                tminf=tmid
                break
            if mc<0:
                break
            xm=xt(tbs[k])
            gm[tbi[k]]=0
            tminf=tbs[k]
        x=xt(tminf)
        g=numpy.matmul(G,x)+d
        onbound=1.*(numpy.abs(x-l)<10.*numpy.spacing(l))-1.*(numpy.abs(x-u)<10.*numpy.spacing(u))
        A=numpy.diag(onbound)
        keep=onbound!=0
        m=numpy.sum(keep)
        A=A[keep,:]
        if m>0:
            Q,R=numpy.linalg.qr(numpy.transpose(A),'complete')
            Z=Q[:,m:]
            ll=numpy.array(l)
            uu=numpy.array(u)
            ll[numpy.isinf(ll)]=-.9*numpy.finfo('d').max
            uu[numpy.isinf(uu)]=.9*numpy.finfo('d').max
            ZGZ=numpy.matmul(numpy.transpose(Z),numpy.matmul(G,Z))
            ulZ=numpy.matmul(numpy.transpose(Z),numpy.column_stack((ll-x,uu-x)))
            lZ=numpy.min(ulZ,1)
            uZ=numpy.max(ulZ,1)
            p=steihaugssquare(-numpy.matmul(numpy.transpose(Z),g),ZGZ,lZ,uZ)
            x=x+numpy.matmul(Z,p)
        else:
            p=steihaugssquare(-g,G,l-x,u-x)
            x=x+p
        g=numpy.matmul(G,x)+d
        if numpy.any(numpy.isinf(x)):
            print('Unbounded')
            return x
        if m>0:
            lambd=numpy.linalg.solve(numpy.matmul(A,numpy.transpose(A)),numpy.matmul(A,g))
            dLdx=g-numpy.matmul(numpy.transpose(A),lambd)
        else:
            dLdx=g
        if numpy.dot(dLdx,dLdx)**.5<tol:
            return x

def steihaugssquare(g,B,l,u):
    A=B
    b=-numpy.array(g)
    tol=1e-8*numpy.dot(g,g)**.5
    n=len(g)
    x=numpy.zeros(n)
    r=-numpy.array(g)
    p=-numpy.array(r)
    rr=numpy.dot(r,r)
    for j in range(n):
        Ap=numpy.matmul(A,p)
        pAp=numpy.dot(p,Ap)
        if pAp<=0:
            return x
        alpha=rr/pAp
        nx=x+alpha*p
        if numpy.any(nx<l-1e-8) or numpy.any(nx>u+1e-8):
            t=numpy.concatenate(((l-x)/p,(u-x)/p))
            tindom=t[(t>=-1e-14)&(t<alpha)]
            if len(tindom)>0:
                t=numpy.min(tindom)
            else:
                t=0
            return x+t*p
        x=nx
        rn=r+alpha*Ap
        rnrn=numpy.dot(rn,rn)
        if rnrn**.5<tol:
            return x
        beta=rnrn/rr
        p=-rn+beta*p
        r=rn
        rr=rnrn
    return x

def symconstraint(f,n,nc):
    xx=sympy.symbols('x0:'+str(n))
    J=sympy.zeros(n,nc);
    if isinstance(f,sympy.MatrixBase):
        for i in range(0,n):
            for j in range(0,nc):
                J[i,j]=sympy.diff(f[j],xx[i])
    else:
        for i in range(0,n):
            for j in range(0,nc):
                J[i,j]=sympy.diff(f(xx)[j],xx[i])
    return J

def symJacobianHessian(f,n):
    xx=sympy.symbols('x0:'+str(n))
    J=sympy.zeros(n,n);
    if isinstance(f,sympy.MatrixBase):
        for i in range(0,n):
            for j in range(0,n):
                J[i,j]=sympy.diff(f[i],xx[j])
    else:
        for i in range(0,n):
            for j in range(0,n):
                J[i,j]=sympy.diff(f(xx)[i],xx[j])
    return J

def symGradient(f,n):
    xx=sympy.symbols('x0:'+str(n))
    G=sympy.zeros(n,1);
    for i in range(0,n):
        G[i]=sympy.diff(f(xx),xx[i])
    return G
def symtolambda(f,g,n,h=[]):
    try:
        xx=sympy.symbols('x0:'+str(n))
        F=sympy.lambdify(xx,f(xx))
        f=lambda x:F(*x)
        if isinstance(g,sympy.MatrixBase):
            G=sympy.lambdify(xx,g)
            g=lambda x:G(*x)[:,0]
        else:
            G=sympy.lambdify(xx,g(xx))
            g=lambda x:G(*x)
    except AttributeError:
        pass
    if h==[]:
        return f,g
    if isinstance(h,sympy.MatrixBase):
        H=sympy.lambdify(xx,h)
    else:
        H=sympy.lambdify(xx,h(xx))
    h=lambda x:numpy.array(H(*x))
    return f,g,h

if __name__ == '__main__':
    #use sympy methods will convert to numpy before evaluating, using sympy to evaluate derivitives if not provided
    f=lambda x:(x[1]-.129*x[0]**2+1.6*x[0]-6)**2+6.07*sympy.cos(x[0])+10
    x=[5.,-8.]
    ce=lambda x:[]
    ci=lambda x:[-(2.*x[0]**2+-x[1]-1.5),-((x[0]-1)**2+(x[1]-2)**2-1.**2)]
    df=lambda x:[2*(-.258*x[0]+1.6)*(x[1]-.129*x[0]**2+1.6*x[0]-6)-6.07*sympy.sin(x[0]),2*(x[1]-.129*x[0]**2+1.6*x[0]-6)]
    ddf=lambda x:[[16641.*x[0]**2./250000.-516.*x[0]/625-129*x[1]/250.+(8./5.-129.*x[0]/500.)*(16./5.-129.*x[0]/250.)-607.*sympy.cos(x[0])/100.+387./125., 16./5.-129.*x[0]/250.],[16./5.-129.*x[0]/250.,2.]]
    dce=lambda x:[]
    ddce=[]
    dci=lambda x:[[-4*x[0],-2*(x[0]-1)],[1,-2*(x[1]-2)]]
    ddci=[lambda x:[[-4,0],[0,0]],lambda x:[[-2,0],[0,-2]]]
    print(sqp(f,x,ce,ci,df,ddf,dce,ddce,dci,ddci))