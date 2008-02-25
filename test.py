#!/usr/bin/env python
from Adolc import *
import numpy as npy

def f(avec):
	a = avec[0]*avec[-1]
	print type(a)
	c =avec[0]*avec[0]
	print type(c)
	return a

ARRAY_LENGTH = 2
x = npy.array([1./(i+1) for i in range(ARRAY_LENGTH)])
ax = npy.array([adouble(0.) for i in range(ARRAY_LENGTH)])
trace_on(1)
for i in range(ARRAY_LENGTH):
	ax[i].is_independent(x[i])# equivalent to ax[i]<<=x[i]
ay = f(ax)
y = depends_on(ay)
trace_off()

y_adolc = function(1,1,x)
py_tape_doc(1,x,npy.array([y]))
