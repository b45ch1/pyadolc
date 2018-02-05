#!/usr/bin/env python

from numpy import *
from pylab import *
import adolc



def explicit_euler(x0,f,ts,p,q):
    N = size(ts)
    if isinstance(p[0],adolc._adolc.adouble):
        x = array([adolc.adouble(0) for m in range(Nm)])
    else:
        x = zeros(N)
    x[0] = x0
    for n in range(1,N):
        h = ts[n] - ts[n-1]
        x[n]= x[n-1] + h*f(ts[n-1],x[n-1],p,q)
    return x


def f(t,x,p,q):
    """ rhs of the ODE"""
    return p[1] + q[0]*x

# analytic solutions
def phi(t,p,q):
    return ((p[1] + q[0]*p[0]) * exp(t/q[0]) - p[1])/q[0]

def phip0(t,p,q):
    return exp(q[0]*t)

def phip1(t,p,q):
    return (exp(q[0]*t)-1.)/q[0]


p = array([10.,2.])
q = array([-1.])
v = concatenate((p,q))

Np = size(p)
Nq = size(q)
Nv = size(v)
Nm = 100
ts = linspace(0,10,Nm)

# TAPING THE INTEGRATOR
adolc.trace_on(1)
av = adolc.adouble(v)
adolc.independent(av)
ax = explicit_euler(av[0],f,ts,av[:Np],av[Np:])
adolc.dependent(ax)
adolc.trace_off()

# COMPUTING FUNCTION AND JACOBIAN FROM THE TAPE
y = adolc.zos_forward(1,v,0)
J = adolc.jacobian(1,v)

x_plot = plot(ts, y,'b')
x_analytical_plot = plot(ts,phi(ts,p,q),'b.')

xp0_plot = plot(ts, J[:,0], 'g')
xp0_analytical_plot = plot(ts, phip0(ts,p,q), 'g.')

xp1_plot = plot(ts, J[:,1], 'r')
xp1_analytical_plot = plot(ts, phip1(ts,p,q), 'r.')

show()
print(J)
