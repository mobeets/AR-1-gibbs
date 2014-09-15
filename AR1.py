
# In[1]:

get_ipython().magic(u'matplotlib inline')


# In[3]:

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma


### Data

# In[245]:

n = 1000.0
alpha = 5.0
beta = 2.0
sig2 = 0.5
X = np.random.normal(0, 1, n)
Y = beta*X + np.random.normal(alpha, np.sqrt(sig2), n)


# In[246]:

plt.plot(Y, 'r-', lw=2, alpha=0.6)
plt.show()


# ## Model
# ### likelihood
#     P(Y | a, b, X) ~ N(a + bX, v)
# 
# ### priors
#     P(a) ~ N(a0, va)
#     P(b) ~ N(b0, vb)
#     P(s) ~ N(v0/2, v0s0/2)
# =>
# 
# ### marginal conditionals
#     P(a | ...) ~ N(ma, Ca)
#     P(b | ...) ~ N(mb, Cb)
#     P(s | ...) ~ IG(v1/2, v1s1/2)

### Priors

# In[247]:

a0 = [0.0, 10.0]
b0 = [0.0, 10.0]
s20 = [3.0, 1.0]


# In[248]:

ap = norm(loc=a0[0], scale=np.sqrt(a0[1]))
bp = norm(loc=b0[0], scale=np.sqrt(b0[1]))
s2p = invgamma(s20[0]/2, loc=0, scale=s20[0]*s20[1]/2)


# In[249]:

x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
plt.plot(x, ap.pdf(x), 'b-', lw=5, alpha=0.6)
plt.title('alpha prior')
plt.show()

plt.plot(x, bp.pdf(x), 'b-', lw=5, alpha=0.6)
plt.title('alpha prior')
plt.show()

x = np.linspace(invgamma.ppf(0.01, s20[0]/2), invgamma.ppf(0.99, v0/2), 100)
plt.plot(x, s2p.pdf(x), 'b-', lw=5, alpha=0.6)
plt.title('sig2 prior')
plt.show()


### Full conditionals

# In[268]:

aC = lambda (a, b, s2): 1/(1/a0[1] + n/s20[1])
am = lambda (a, b, s2): aC((a,b,s2))*(sum(Y-b*X)/s20[1] + a0[0]/a0[1])
bC = lambda (a, b, s2): 1/(1/b0[1] + sum(X**2)/s20[1])
bm = lambda (a, b, s2): bC((a,b,s2))*(sum((Y-a)*X)/s20[1]+b0[0]/b0[1])
sA = lambda (a, b, s2): s20[0] + n
sB = lambda (a, b, s2): s20[0]*s20[1] + sum((Y-a-b*X)**2)


# In[269]:

ar = lambda theta: norm(loc=am(theta), scale=np.sqrt(aC(theta)))
br = lambda theta: norm(loc=bm(theta), scale=np.sqrt(bC(theta)))
sr = lambda theta: invgamma(sA(theta)/2, loc=0, scale=sB(theta)/2)


### Gibbs Sampling

# In[270]:

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


# In[271]:

x1 = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
x2 = np.linspace(invgamma.ppf(0.01, v0/2), invgamma.ppf(0.99, v0/2), 100)

plt.plot(x1, ac(tMAP).pdf(x), 'r-', lw=5, alpha=0.6, label='a prior')
plt.title('alpha conditional at MAP')
plt.show()

plt.plot(x1, bc(tMAP).pdf(x), 'r-', lw=5, alpha=0.6, label='b prior')
plt.title('beta conditional at MAP')
plt.show()

plt.plot(x2, sc(tMAP).pdf(x), 'r-', lw=5, alpha=0.6, label='sig2 prior')
plt.title('sig2 conditional at MAP')
plt.show()


# In[272]:

plt.plot(ts[:,0], '-')
plt.title('alpha samples')
plt.show()
plt.plot(ts[:,1], '-')
plt.title('beta samples')
plt.show()
plt.plot(ts[:,2], '-')
plt.title('sig2 samples')
plt.show()


# In[273]:

plt.hist(ts[:,0], normed=True, histtype='stepfilled', alpha=0.2)
plt.title('alpha posterior')
plt.show()
plt.hist(ts[:,1], normed=True, histtype='stepfilled', alpha=0.2)
plt.title('beta posterior')
plt.show()
plt.hist(ts[:,2], normed=True, histtype='stepfilled', alpha=0.2)
plt.title('sig2 posterior')
plt.show()


# In[261]:



