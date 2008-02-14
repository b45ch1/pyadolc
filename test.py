#!/usr/bin/env python

import Adolc

print 'computing square of 4 = ',Adolc.square(4)
print 'computing the max of 1 and 2 = ',Adolc.fmax(1,2)

a = Adolc.adouble(13.)
b = Adolc.adouble(5.)
c = Adolc.adouble(7.)

Adolc.myprintf(c)

print 'a=',a
print 'b=',b
print 'c=',c

a *= b
print 'a=',a

#c = a
#print 'c=', c


c = a * b

print '%s=%s*%s'%(c,a,b)



def speelpenning(avec):
	tmp = 1.
	for a in avec:
		tmp *= a
	return tmp


import numpy as npy
myavec = npy.array([Adolc.adouble(i) for i in range(1,10)])
print myavec

print speelpenning(myavec)