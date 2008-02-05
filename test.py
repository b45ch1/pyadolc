#!/usr/bin/env python

import Adolc

print Adolc.square(4)
print Adolc.fmax(1,2)

a = Adolc.adouble(1.)
b = Adolc.adouble(2.)
c = Adolc.adouble(3.)

print 'a=',a
print 'b=',b
print 'c=',c

a *= b
print 'a=',a

c = a
print 'c=', c


a * b
