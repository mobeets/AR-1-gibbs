import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma

def plot1(D):
    plt.plot(D.X, D.Y, 'k.', lw=2, alpha=0.6)
    plt.xlabel('X(t)')
    plt.ylabel('Y(t)')
    plt.show()

    plt.plot(D.X, 'k-', lw=2, alpha=0.6)
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.show()

    plt.plot(D.Y, 'k-', lw=2, alpha=0.6)
    plt.xlabel('t')
    plt.ylabel('Y(t)')
    plt.show()


def plot2(A,B,S,T):
    x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    plt.plot(x, A.prior.rv.pdf(x), 'b-', lw=5, alpha=0.6)
    plt.title('alpha prior')
    plt.show()

    plt.plot(x, B.prior.rv.pdf(x), 'b-', lw=5, alpha=0.6)
    plt.title('beta prior')
    plt.show()

    x = np.linspace(invgamma.ppf(0.01, S.prior.p1/2), invgamma.ppf(0.99, S.prior.p1/2), 100)
    plt.plot(x, S.prior.rv.pdf(x), 'b-', lw=5, alpha=0.6)
    plt.title('sig2 prior')
    plt.show()

    x = np.linspace(invgamma.ppf(0.01, T.prior.p1/2), invgamma.ppf(0.99, T.prior.p1/2), 100)
    plt.plot(x, T.prior.rv.pdf(x), 'b-', lw=5, alpha=0.6)
    plt.title('omg2 prior')
    plt.show()


def plot3(A,B,S,T,D, tMAP=None):
    States = D.Y
    pMAP = [9.75961269, 0.0583687877, 78.4901534, 78.1696975]
    alpha, beta, sig2, ome2 = pMAP
    tMAP = (alpha, beta, sig2, ome2, States, D.Y)

    x1 = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    x2 = np.linspace(invgamma.ppf(0.01, S.prior.p1/2), invgamma.ppf(0.99, S.prior.p1/2), 100)
    x3 = np.linspace(invgamma.ppf(0.01, T.prior.p1/2), invgamma.ppf(0.99, T.prior.p1/2), 100)

    tmp = lambda (a, b, s, t, X, Y): 1/(1/A.prior.var + D.n/S.prior.p2)
    print tMAP
    print tmp(tMAP)
    print A.cond.var(tMAP)
    print A.cond.m(tMAP)
    print A.cond.rv((tMAP))

    plt.plot(x1, A.cond.rv(tMAP).pdf(x1), 'r-', lw=5, alpha=0.6, label='a prior')
    plt.title('alpha conditional at MAP')
    plt.show()

    plt.plot(x1, B.cond.rv(tMAP).pdf(x1), 'r-', lw=5, alpha=0.6, label='b prior')
    plt.title('beta conditional at MAP')
    plt.show()

    plt.plot(x2, S.cond.rv(tMAP).pdf(x2), 'r-', lw=5, alpha=0.6, label='sig2 prior')
    plt.title('sig2 conditional at MAP')
    plt.show()

    plt.plot(x3, T.cond.rv(tMAP).pdf(x3), 'r-', lw=5, alpha=0.6, label='sig2 prior')
    plt.title('sig2 conditional at MAP')
    plt.show()


def plot4(ts):
    plt.plot(ts[:,0], 'r-')
    plt.title('alpha samples')
    plt.show()
    plt.plot(ts[:,1], 'r-')
    plt.title('beta samples')
    plt.show()
    plt.plot(ts[:,2], 'r-')
    plt.title('sig2 samples')
    plt.show()
    plt.plot(ts[:,3], 'r-')
    plt.title('ome2 samples')
    plt.show()

def plot5(ts):
    plt.hist(ts[:,0], normed=True, histtype='stepfilled', alpha=0.2)
    plt.title('alpha posterior')
    plt.show()
    plt.hist(ts[:,1], normed=True, histtype='stepfilled', alpha=0.2)
    plt.title('beta posterior')
    plt.show()
    plt.hist(ts[:,2], normed=True, histtype='stepfilled', alpha=0.2)
    plt.title('sig2 posterior')
    plt.show()
    plt.hist(ts[:,3], normed=True, histtype='stepfilled', alpha=0.2)
    plt.title('ome2 posterior')
    plt.show()
