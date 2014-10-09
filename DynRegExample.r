MKT = scan("MKTindex.txt")
Y   = scan("AAPL.txt")

Y = Y - mean(Y)
M0 = lm(Y~MKT-1)


######
# Let's first run a dynamic regression using the 
# first order DLM
#
#  Y_t = b_t X_t + eps_t
#  b_t = b_t-1 + w_t 
#  We will use the closed-form solution with discount factor delta 
##### 


T = length(Y)

delta = 0.95
m = rep(1,T+1)
C = rep(1,T+1)

n = rep(5,T+1)
d = rep((5*20),T+1)

S = d[1]/n[1]

for(i in 2:(T+1)){
	
	R = C[i-1]/delta
	f = MKT[i-1]*m[i-1]
	Q = R*MKT[i-1]^2 + S
	e = Y[i-1] - f
		
	n[i] = n[i-1] + 1
	
	d[i] = d[i-1] + S*(e^2)/Q
	
	A = MKT[i-1]*R/Q
	m[i] = m[i-1] + A*e
	
	S = d[i]/n[i] 
	C[i] = R*S/Q 
	}

ts.plot(m,lwd=2,col=4,main="Posterior Mean (beta_t|D_t)")
abline(h=M0$coef[1],col=2,lwd=2)

ts.plot(d/n,lwd=2,col=4,main="Posterior summary (sigma|D_t)")
abline(h=summary(M0)$sigma^2,col=2,lwd=2)



#####
# Alternatively...
#  Y_t = b_t X_t + eps_t (sigma)
#  b_t = alpha + beta b_t-1 + w_t (omega) 
#  We will use a MCMC 
##### 
set.seed(1243)


# Definition of FFBS function
#-------------------------------------------------------
# Univariate FFBS: 
# y(t)     ~ N(alpha(t)+F(t)*theta(t);V(t))        
# theta(t) ~ N(gamma+G*theta(t-1);W)               
#-------------------------------------------------------
ffbsu = function(y,F,alpha,V,G,gamma,W,a1,R1,nd=1){
  n = length(y)
  if (length(F)==1){F = rep(F,n)}
  if (length(alpha)==1){alpha = rep(alpha,n)}
  if (length(V)==1){V = rep(V,n)}
  a = rep(0,n)
  R = rep(0,n)
  m = rep(0,n)
  C = rep(0,n)
  B = rep(0,n-1)
  H = rep(0,n-1)
  # time t=1
  a[1] = a1
  R[1] = R1
  f    = alpha[1]+F[1]*a[1]
  Q    = R[1]*F[1]**2+V[1]
  A    = R[1]*F[1]/Q
  m[1] = a[1]+A*(y[1]-f)
  C[1] = R[1]-Q*A**2
  # forward filtering
  for (t in 2:n){
    a[t] = gamma + G*m[t-1]
    R[t] = C[t-1]*G**2 + W
    f    = alpha[t]+F[t]*a[t]
    Q    = R[t]*F[t]**2+V[t]
    A    = R[t]*F[t]/Q
    m[t] = a[t]+A*(y[t]-f)
    C[t] = R[t]-Q*A**2
    B[t-1] = C[t-1]*G/R[t]
    H[t-1] = sqrt(C[t-1]-R[t]*B[t-1]**2)
  }
  # backward sampling
  theta = matrix(0,nd,n)
  theta[,n] = rnorm(nd,m[n],sqrt(C[n]))
  for (t in (n-1):1)
    theta[,t] = rnorm(nd,m[t]+B[t]*(theta[,t+1]-a[t+1]),H[t])
  if (nd==1){
    theta[1,]
  }
  else{
    theta
  }
}

# Function for the variances
#-------------------------------------------------------
# y    = X + u    u ~ N(0,sig2*I_n)
#
#      sig2 ~ IG(v/2,v*lam/2)
#-------------------------------------------------------
fixedparSIG = function(y,X,v,lam){
  n     = length(y)
  par1  = (v+n)/2
  par2  = v*lam + sum((y-X)^2)
  sig2  = 1/rgamma(1,par1,par2/2)
  return(sig2)
}





T = length(Y)


v.sig     = 5 
lam.sig   = 20

v.omega   = 10
lam.omega = 0.1^2

## Initial values for DLM
sigma2= 15
omega2= 0.1^2

M         = 1000

BETAS    = matrix(0,M,T)
PARS     = matrix(0,M,2)

m0 = 1
C0 = 3

for(iter in 1:M){
  	
    aux = ffbsu(Y,MKT,0,sigma2,1,0,omega2,m0,C0)
    
	sigma2  = fixedparSIG(Y,aux*MKT,v.sig,lam.sig)  
    
    omega2 = fixedparSIG(aux[2:T],aux[1:(T-1)],v.omega,lam.omega)         
    
    BETAS[iter,] = aux
    PARS[iter,1] = sigma2; PARS[iter,2] = omega2; 
	print(iter)
}

par(mfrow=c(2,2))
ts.plot(PARS[,1],col=2,lwd=1,ylab="Sigma2")
hist(PARS[,1],prob=T,col=2,main="Sigma2")
ts.plot(PARS[,2],col=4,lwd=1,ylab="Omega2")
hist(PARS[,2],prob=T,col=4,main="Omega2")

mBeta = apply(BETAS[500:1000,],2,mean)
q5Beta = apply(BETAS[500:1000,],2,quant05)
q95Beta = apply(BETAS[500:1000,],2,quant95)

par(mfrow=c(1,1))
ts.plot(mBeta,col=4,lwd=2)
abline(h=M0$coef[1],col=2,lwd=2)

par(mfrow=c(1,1))
ts.plot(mBeta,col=4,lwd=1,ylim=range(q5Beta,q95Beta))
abline(h=M0$coef[1],col=2,lwd=2)
lines(q5Beta,col=2,lty=2)
lines(q95Beta,col=2,lty=2)
