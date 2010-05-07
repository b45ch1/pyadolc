"""
Compare evaluation of derivatives with Sympy and with PYADOLC at the example function

def f(x):
    retval = 0.
    for n in range(1,N):
        for m in range(n):
            retval += 1./ norm(x[n,:] - x[m,:],2)
    return retval

"""

import sympy
import adolc
import numpy
from numpy import array, zeros, ones, shape
from numpy.random import random
from numpy.linalg import norm


# setup problem size
N, D, = 2,2
M = N + 3

################################
# PART 0: by hand
################################


def f(x):
    retval = 0.
    for n in range(1,N):
        for m in range(n):
            retval += 1./ norm(x[n,:] - x[m,:],2)
    return retval

def df(x):
    g = zeros(shape(x),dtype=float)
    for n in range(N):
        for d in range(D):
            for m in range(N):
                if n != m:
                    g[n,d] -= (x[n,d] - x[m,d])/norm(x[n,:]-x[m,:])**3
    return g

def ddf(x):
    N,D = shape(x)
    H = zeros((N,D,N,D),dtype=float)
    for n in range(N):
        for d in range(D):
            for m in range(N):
                for e in range(D):
                    for l in range(N):
                        if l==n:
                            continue
                        H[n,d,m,e] -= (( (m==n) * (d==e) - (m==l)*(d==e) ) - 3* (x[n,d] - x[l,d])/norm(x[n,:]-x[l,:])**2 * ( (n==m) - (m==l))*( x[n,e] - x[l,e]))/norm(x[n,:] - x[l,:])**3
    return H


################################
# PART 1: computation with SYMPY
################################

xs = array([[sympy.Symbol('x%d%d'%(n,d)) for d in range(D)] for n in range(N)])
# computing the function f: R^(NxD) -> R symbolically
fs = 0
for n in range(1,N):
    for m in range(n):
        tmp = 0
        for d in range(D):
            tmp += (xs[n,d] - xs[m,d])**2
        tmp = sympy.sqrt(tmp)
        fs += 1/tmp

# computing the gradient symbolically
dfs = array([[sympy.diff(fs, xs[n,d]) for d in range(D)] for n in range(N)])

# computing the Hessian symbolically
ddfs = array([[[[ sympy.diff(dfs[m,e], xs[n,d]) for d in range(D)] for n in range(N)] for e in range(D) ] for m in range(N)])

def sym_f(x):
    symdict = dict()
    for n in range(N):
        for d in range(D):
            symdict[xs[n,d]] = x[n,d]
    return fs.subs(symdict).evalf()

def sym_df(x):
    symdict = dict()
    for n in range(N):
        for d in range(D):
            symdict[xs[n,d]] = x[n,d]
    return array([[dfs[n,d].subs(symdict).evalf() for d in range(D)] for n in range(N)])

def sym_ddf(x):
    symdict = dict()
    for n in range(N):
        for d in range(D):
            symdict[xs[n,d]] = x[n,d]
    return array([[[[ ddfs[m,e,n,d].subs(symdict).evalf() for d in range(D)] for n in range(N)] for e in range(D)] for m in range(N)],dtype=float)

###################################
# PART 1: computation with PYADOLC
###################################

adolc.trace_on(0)
x = adolc.adouble(numpy.random.rand(*(N,D)))
adolc.independent(x)
y = f(x)
adolc.dependent(y)
adolc.trace_off()


# point at which the derivatives should be evaluated
x = random((N,D))

print '\n\n'
print 'Sympy function = function  check (should be almost zero)'
print f(x) - sym_f(x)

print '\n\n'
print 'Sympy vs Hand Derived Gradient check (should be almost zero)'
print df(x) - sym_df(x)

print 'Sympy vs Ad Derived Gradient check (should be almost zero)'
print adolc.gradient(0, numpy.ravel(x)).reshape(x.shape) - sym_df(x)

print '\n\n'
print 'Sympy vs Hand Derived Hessian check (should be almost zero)'
print ddf(x) - sym_ddf(x)

print 'Sympy vs Ad Derive Hessian check (should be almost zero)'
print adolc.hessian(0, numpy.ravel(x)).reshape(x.shape + x.shape) - sym_ddf(x)




