#!/usr/bin/env python
from adolc import *
import numpy as npy

N = 10000000
#x = npy.array([1./(i+1) for i in range(N)])
#ax = npy.array([adouble(0.) for i in range(N)])
#print 'trace_on'
#trace_on(10)
#for i in range(N):
	#ax[i].is_independent(x[i])# equivalent to ax[i]<<=x[i]
#print 'first assignment'
#a = 1.
#for i in range(N):
	#a=a * ax[i]
##b = ax[0]*ax[1]*ax[2]*ax[3]*ax[4]*ax[5]*ax[6]
#y = depends_on(a)
#trace_off()

#y_adolc = function(10,x)
#print 'result=', y_adolc
##tape_to_latex(10,x,npy.array([y]))


ax = adouble(0.)
x = 1


trace_on(10)
ax.is_independent(x)
ay = ax
for i in range(N):
	ay = ay * ay
depends_on(ay)
trace_off()