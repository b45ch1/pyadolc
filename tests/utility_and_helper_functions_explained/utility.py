#!/usr/bin/env python
from adolc import *
import numpy

def f(avec):
	a = avec[0]*avec[-1]
	c =avec[0]*avec[0]
	if isinstance(a,badouble):
		print avec[0].location
		print avec[1].location
		print a.location
		print c.location
	return a


N = 2
x = numpy.array([1.,2.])
ax = numpy.array([adouble(0.),adouble(0.)])

trace_on(1)
for n in range(N):
	ax[n].is_independent(x[n])
ay = f(ax)
depends_on(ay)
trace_off()

tape_to_latex(1,x,numpy.array([0]))
