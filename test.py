#!/usr/bin/env python

import Adolc


a = Adolc.adouble(13.)
b = Adolc.adouble(5.)
c = Adolc.adouble(7.)


print 'a=',a
print 'b=',b
print 'c=',c

a *= b
print 'a=',a

#c = a
#print 'c=', c


c = a * b

print '%s=%s*%s'%(c,a,b)

print type(c)


def speelpenning(avec):
	tmp = Adolc.adouble(4.)
	for a in avec:
		tmp *= a
	return tmp


import numpy as npy
ax = npy.array([Adolc.adouble(i) for i in range(1,10)])
x = npy.array([1. + npy.exp(-i) for i in range(1,10)])
y = 0.
Adolc.trace_on(1)
for i in range(npy.shape(ax)[0]):
	ax[i]<<=x[i]
ay = speelpenning(ax)
#ay >>= y
y = Adolc.depends_on(ay)
Adolc.trace_off()

#print y
#print x

asdf = Adolc.function(1,1,x)
print 'asdf=',asdf
#g = Adolc.gradient(1,x)

#print 'gradient is', g
