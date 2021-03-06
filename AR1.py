
# In[3]:

get_ipython().magic(u'matplotlib inline')


# In[4]:

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma


# In[5]:

class Params:
    def __init__(self):
        pass


### Data

# In[134]:

D = Params()
D.n = 1000
D.alpha = 5.0
D.beta = 0.5

# D.Vv = 4.0
D.Wv = 6.0

# D.V = np.random.normal(0, np.sqrt(D.Vv), D.n)
D.W = np.random.normal(D.alpha, np.sqrt(D.Wv), D.n)
D.X = np.zeros(D.n+1)
D.X[0] = 0.0
for i in xrange(D.n):
    D.X[i+1] = D.beta*D.X[i] + D.W[i]
D.Y = D.X[1:]# + D.V
D.X = D.X[:-1]


# In[135]:

plt.plot(D.X, D.Y, 'k.', lw=2, alpha=0.6)
plt.xlabel('x(t-1)')
plt.ylabel('x(t)')
plt.show()

plt.plot(D.X, 'k-', lw=2, alpha=0.6)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()


### Priors

# In[136]:

a0 = [0.0, 10.0]
b0 = [0.0, 10.0]
s20 = [3.0, 1.0]


# In[137]:

ap = norm(loc=a0[0], scale=np.sqrt(a0[1]))
bp = norm(loc=b0[0], scale=np.sqrt(b0[1]))
s2p = invgamma(s20[0]/2, loc=0, scale=s20[0]*s20[1]/2)


# In[138]:

x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
plt.plot(x, ap.pdf(x), 'b-', lw=5, alpha=0.6)
plt.title('alpha prior')
plt.show()

plt.plot(x, bp.pdf(x), 'b-', lw=5, alpha=0.6)
plt.title('beta prior')
plt.show()

x = np.linspace(invgamma.ppf(0.01, s20[0]/2), invgamma.ppf(0.99, s20[1]/2), 100)
plt.plot(x, s2p.pdf(x), 'b-', lw=5, alpha=0.6)
plt.title('sig2 prior')
plt.show()


### Full conditionals

# In[139]:

aC = lambda (a, b, s2): 1/(1/a0[1] + D.n/s20[1])
am = lambda (a, b, s2): aC((a,b,s2))*(sum(D.Y-b*D.X)/s20[1] + a0[0]/a0[1])
bC = lambda (a, b, s2): 1/(1/b0[1] + sum(D.X**2)/s20[1])
bm = lambda (a, b, s2): bC((a,b,s2))*(sum((D.Y-a)*D.X)/s20[1]+b0[0]/b0[1])
sA = lambda (a, b, s2): s20[0] + D.n
sB = lambda (a, b, s2): s20[0]*s20[1] + sum((D.Y-a-b*D.X)**2)


# In[140]:

ar = lambda theta: norm(loc=am(theta), scale=np.sqrt(aC(theta)))
br = lambda theta: norm(loc=bm(theta), scale=np.sqrt(bC(theta)))
sr = lambda theta: invgamma(sA(theta)/2, loc=0, scale=sB(theta)/2)


### Gibbs Sampling

# In[141]:

aCur, bCur, sCur = a0[0], b0[0], s20[0]
niters = 1000
ts = np.zeros([niters, 3])
for i in xrange(niters):
    aCur = ar((aCur, bCur, sCur)).rvs(size=1)
    bCur = br((aCur, bCur, sCur)).rvs(size=1)
    sCur = sr((aCur, bCur, sCur)).rvs(size=1)
    ts[i] = (aCur, bCur, sCur)
tMAP = ts.mean(0)
print '(alpha, beta, sig2) = {0}'.format(tMAP)


# In[142]:

x1 = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
x2 = np.linspace(invgamma.ppf(0.01, s20[0]/2), invgamma.ppf(0.99, s20[0]/2), 100)

plt.plot(x1, ar(tMAP).pdf(x1), 'r-', lw=5, alpha=0.6, label='a prior')
plt.title('alpha conditional at MAP')
plt.show()

plt.plot(x1, br(tMAP).pdf(x1), 'r-', lw=5, alpha=0.6, label='b prior')
plt.title('beta conditional at MAP')
plt.show()

plt.plot(x2, sr(tMAP).pdf(x2), 'r-', lw=5, alpha=0.6, label='sig2 prior')
plt.title('sig2 conditional at MAP')
plt.show()


# In[143]:

plt.plot(ts[:,0], 'r-')
plt.title('alpha samples')
plt.show()
plt.plot(ts[:,1], 'r-')
plt.title('beta samples')
plt.show()
plt.plot(ts[:,2], 'r-')
plt.title('sig2 samples')
plt.show()


# In[144]:

plt.hist(ts[:,0], normed=True, histtype='stepfilled', alpha=0.2)
plt.title('alpha posterior')
plt.show()
plt.hist(ts[:,1], normed=True, histtype='stepfilled', alpha=0.2)
plt.title('beta posterior')
plt.show()
plt.hist(ts[:,2], normed=True, histtype='stepfilled', alpha=0.2)
plt.title('sig2 posterior')
plt.show()


# In[144]:



