import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma

from plots import *

class Params:
    pass

class ParamsHolder:
    def __init__(self, A, B, S, T):
        self.A = A
        self.B = B
        self.S = S
        self.T = T

    def draw_theta(self, (aCur, bCur, sCur, tCur), X, Y):
        aCur = self.A.cond.rv((aCur, bCur, sCur, tCur, X, Y)).rvs(size=1)
        bCur = self.B.cond.rv((aCur, bCur, sCur, tCur, X, Y)).rvs(size=1)
        sCur = self.S.cond.rv((aCur, bCur, sCur, tCur, X, Y)).rvs(size=1)
        tCur = self.T.cond.rv((aCur, bCur, sCur, tCur, X, Y)).rvs(size=1)
        return (aCur, bCur, sCur, tCur)

def data():
    D = Params()
    D.n = 1000
    D.alpha = 5.0
    D.beta = 0.5

    D.Vv = 4.0
    D.Wv = 6.0

    D.V = np.random.normal(0, np.sqrt(D.Vv), D.n)
    D.W = np.random.normal(D.alpha, np.sqrt(D.Wv), D.n)
    D.X = np.zeros(D.n+1)
    for i in xrange(D.n):
        D.X[i+1] = D.beta*D.X[i] + D.W[i]
    D.Y = D.X[1:] + D.V
    D.X = D.X[1:]
    return D

def params(D):
    A = Params()
    A.prior = Params()
    B = Params()
    B.prior = Params()
    S = Params()
    S.prior = Params()
    T = Params()
    T.prior = Params()

    (A.prior.m, A.prior.var) = (0.0, 10.0)
    (B.prior.m, B.prior.var) = (0.0, 10.0)
    (S.prior.p1, S.prior.p2) = (3.0, 1.0)
    (T.prior.p1, T.prior.p2) = (3.0, 1.0)

    A.prior.rv = norm(loc=A.prior.m, scale=np.sqrt(A.prior.var))
    B.prior.rv = norm(loc=B.prior.m, scale=np.sqrt(A.prior.var))
    S.prior.rv = invgamma(S.prior.p1/2, loc=0, scale=S.prior.p1*S.prior.p2/2)
    T.prior.rv = invgamma(T.prior.p1/2, loc=0, scale=T.prior.p1*T.prior.p2/2)

    ### Full conditionals

    A.cond = Params()
    B.cond = Params()
    S.cond = Params()
    T.cond = Params()

    # AR-1 without noise; regress X[1:] on X[:-1]
    A.cond.var = lambda (a, b, s, t, X, Y): 1/(1/A.prior.var + D.n/S.prior.p2)
    A.cond.m = lambda (a, b, s, t, X, Y): A.cond.var((a,b,s,t,X,Y))*(sum(X[1:]-b*X[:-1])/S.prior.p2 + A.prior.m/A.prior.var)
    B.cond.var = lambda (a, b, s, t, X, Y): 1/(1/B.prior.var + sum(X**2)/S.prior.p2)
    B.cond.m = lambda (a, b, s, t, X, Y): B.cond.var((a,b,s,t,X,Y))*(sum((X[1:]-a)*X[:-1])/S.prior.p2 + B.prior.m/B.prior.var)
    S.cond.a = lambda (a, b, s, t, X, Y): S.prior.p1 + D.n
    S.cond.b = lambda (a, b, s, t, X, Y): S.prior.p1*S.prior.p2 + sum((X[1:]-a-b*X[:-1])**2)

    # Regression of Y on X
    T.cond.a = lambda (a, b, s, t, X, Y): T.prior.p1 + D.n
    T.cond.b = lambda (a, b, s, t, X, Y): T.prior.p1*T.prior.p2 + sum((Y-X)**2)

    # conditional draws
    A.cond.rv = lambda theta: norm(loc=A.cond.m(theta), scale=np.sqrt(A.cond.var(theta)))
    B.cond.rv = lambda theta: norm(loc=B.cond.m(theta), scale=np.sqrt(B.cond.var(theta)))
    S.cond.rv = lambda theta: invgamma(S.cond.a(theta)/2, loc=0, scale=S.cond.b(theta)/2)
    T.cond.rv = lambda theta: invgamma(T.cond.a(theta)/2, loc=0, scale=T.cond.b(theta)/2)

    return ParamsHolder(A, B, S, T)

def inner_backward((a,b,s,t), Xt1, Yt1, mt, Ct, rt):
    """
    n.b. read Xt1 as "X_{t+1}"
    """
    ht = mt + (b* Ct * (Xt1 - rt))/Yt1
    Ht = Ct - (b**2)*(Ct**2)/Yt1
    return norm(loc=ht, scale=np.sqrt(Ht)).rvs(size=1)

def backward((a,b,s,t), XT, Y, m, C, r):
    """
    generates each Xt, given each X_{t+1}
    """
    X = np.zeros(len(Y))
    X[-1] = XT
    for i in xrange(len(Y)-2, -1, -1):
        X[i] = inner_backward((a,b,s,t), X[i+1], Y[i+1], m[i], C[i], r[i])
    return X

def forward((a,b,s,t), Y):
    """
    generates the last X, and parameters for its distribution N(m,C)
    """
    # initialize?
    m = np.zeros(len(Y))
    C = np.zeros(len(Y))
    r = np.zeros(len(Y))
    R = np.zeros(len(Y))
    f = np.zeros(len(Y))
    F = np.zeros(len(Y))
    h = np.zeros(len(Y))
    H = np.zeros(len(Y))
    for i in xrange(len(Y)):
        # prior at i-1 ~ N(r[i], R[i])
        r[i] = a + b*m[i-1]
        R[i] = (b**2)*C[i] + t
        # predictive at i-1 ~ N(f[i], F[i])
        f[i] = r[i]
        F[i] = R[i] + s
        # posterior at i ~ N(m[i], C[i])
        h[i] = Y[i] - f[i]
        H[i] = R[i]/F[i]
        m[i] = r[i] + H[i]*h[i]
        C[i] = R[i] - R[i]*H[i]
    XT = norm(loc=m[-1], scale=np.sqrt(C[-1])).rvs(size=1)
    return XT, m, C, r

def ffbs(theta, Y):
    XT, m, C, r = forward(theta, Y)
    X = backward(theta, XT, Y, m, C, r)
    return X

def gibbs(D, P):
    niters = 100
    Thetas = np.zeros([niters+1, 4])
    States = np.zeros([niters+1, len(D.Y)])
    States[0] = D.Y
    for i in xrange(1, niters+1):
        Thetas[i] = P.draw_theta(Thetas[i-1], States[i-1], D.Y)
        States[i] = ffbs(Thetas[i], D.Y)
    tMAP = Thetas.mean(0)
    print '(alpha, beta, sig2, ome2) = {0}'.format(tMAP)

def main():
    D = data()
    P = params(D)
    plot3(P.A, P.B, P.S, P.T, D)
    # gibbs(D, P)

if __name__ == '__main__':
    main()
