#!/usr/bin/env python
from Adolc import *

## constructors
#a = adouble(13.);	print 'a=adouble(13.)\t= ',a,'\t\ta.val =',a.val
#b = adouble(5);		print 'b=adouble(5)\t= ',b,'\t\tb.val =',b.val


## unary
#print '-a  \t =',-a
#print '+a  \t =',+a

## operator + for int and double
#print 'a+2  \t =',a+2
#print 'a+2. \t =',a+2.
#print '2+a  \t =',2+a
#print '2.+a \t =',2.+a

## operator - for int and double
#print 'a-2  \t =',a-2
#print 'a-2. \t =',a-2.
#print '2-a  \t =',2-a
#print '2.-a \t =',2.-a

## operator * for int and double
#print 'a*2  \t =',a*2
#print 'a*2. \t =',a*2.
#print '2*a  \t =',2*a
#print '2.*a \t =',2.*a

## operator / for int and double
#print 'a/2  \t =',a/2
#print 'a/2. \t =',a/2.
#print '2/a  \t =',2/a
#print '2./a \t =',2./a

##operator +,-,*,/ for badouble
#print 'a+b  \t =',a+b
#print 'a-b  \t =',a-b
#print 'a*b  \t =',a*b
#print 'a/b  \t =',a/b

## operator +=,-=,*=,/= for badouble
#a+=b; print 'a+=b  \t =',a
#a-=b; print 'a-=b  \t =',a
#a*=b; print 'a*=b  \t =',a
#a/=b; print 'a/=b  \t =',a

## operator **
#print 'a**2.', a**2.
#print 'a**2', a**2
#print "to be implemented print '2.**a', 2.**a"

##functions
#import numpy as npy
#a = adouble(0.4);	print 'a=adouble(13.)\t= ',a,'\t\ta.val =',a.val
#print 'exp  (a)', npy.exp  (a), a.exp  ()
#print 'log  (a)', npy.log  (a), a.log  ()
#print 'sqrt (a)', npy.sqrt (a), a.sqrt ()
#print 'sin  (a)', npy.sin  (a), a.sin  ()
#print 'cos  (a)', npy.cos  (a), a.cos  ()
#print 'tan  (a)', npy.tan  (a), a.tan  ()
#print 'asin (a)', npy.arcsin(a),a.asin ()
#print 'acos (a)', npy.arccos(a),a.acos ()
#print 'atan (a)', npy.arctan(a),a.atan ()
#print 'log10(a)', npy.log10(a), a.log10()
#print 'sinh (a)', npy.sinh (a), a.sinh ()
#print 'cosh (a)', npy.cosh (a), a.cosh ()
#print 'tanh (a)', npy.tanh (a), a.tanh ()
#print 'fabs (a)', npy.fabs (a), a.fabs ()
#print 'ceil (a)', npy.ceil (a), a.ceil ()
#print 'floor (a)',npy.floor (a),a.floor ()
#print 'fmax (a,b)',			    a.fmax (b)
#print 'fmax (a,0.3)',			a.fmax (0.3)
#print 'fmax (0.3,a)','/* not implemented */'
#print 'fmin (a,b)',			    a.fmin (b)
#print 'fmin (a,0.3)',			a.fmin (0.3)
#print 'fmin (0.3,a)','/* not implemented */'


# SETUP FOR ALL TESTS
# ------------------

ARRAY_LENGTH = 2
import numpy as npy
import time 		# to make a runtime analysis
x = npy.array([1./(i+1) for i in range(ARRAY_LENGTH)])



direction = npy.zeros(ARRAY_LENGTH)
direction[0] = 1.

def run_test(f,x_0,direction, message='', print_derivatives=False):
	print message
	ax = npy.array([adouble(0.) for i in range(ARRAY_LENGTH)])

	#for i in range(ARRAY_LENGTH):
		#print ax[i],'\t',ax[i].__repr__(), ax[i].location

	
	trace_on(1)
	for i in range(ARRAY_LENGTH):
		ax[i].is_independent(x[i])# equivalent to ax[i]<<=x[i]
	ay = f(ax)
	y = depends_on(ay)
	trace_off()

	start_time = time.time()
	y_adolc = function(1,1,x)
	run_time_adolc = time.time() - start_time

	start_time = time.time()
	y_normal = f(x)
	run_time_normal = time.time() - start_time

	start_time = time.time()
	g = gradient(1,x)
	run_time_gradient = time.time() - start_time

	start_time = time.time()
	y_and_deriv = fos_forward(1,1,1,x,direction)
	run_time_fos_forward = time.time() - start_time

	print 'Adolc\tfunction evaluation:\t%f\telapsed time: %f'%(y_adolc,run_time_adolc)
	print 'normal\tfunction evaluation:\t%f\telapsed time: %f'%(y_normal,run_time_normal)
	print 'gradient evaluation:\t\t........\telapsed time: %f'%run_time_gradient,'\t',
	if print_derivatives == True:
		 print g
	else:
		 print ''
	print 'direct. diff. evaluation:\t%f\telapsed time: %f'%(y_and_deriv['y'],run_time_fos_forward),
	if print_derivatives == True:
		 print  y_and_deriv['directional_derivative']
	else:
		 print ''
	print ''
	return y_normal


def f(avec):
	a = avec[0]*avec[-1]
	c =avec[0]*avec[0]
	if isinstance(a,badouble):
		print avec[0].location
		print avec[1].location
		print a.location
		print c.location
	return a
y = run_test(f,x,direction)


py_tape_doc(1,x,npy.array([y]))

#def f(avec):
	#return npy.sum(avec)
#run_test(f,x,direction)

#def f(avec):
	#return npy.prod(avec)
#run_test(f,x,direction)

#def f(avec):
	#import numpy as npy
	#N = npy.shape(avec)[0]
	#A = 3.*npy.eye(N)
	#y = npy.dot(A,avec)
	#return npy.dot(avec,y)
#run_test(f,x,direction,'inner product')

#def f(avec):
	#import numpy as npy
	#N = npy.shape(avec)[0]
	#A = npy.eye(N) - 2* npy.outer(avec,avec)
	#y = npy.dot(avec,npy.dot(A,avec))
	#return y
#run_test(f,x,direction)
