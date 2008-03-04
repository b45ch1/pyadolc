import numpy as npy
from Adolc import *


# SETUP FOR ALL TESTS
# ------------------

ARRAY_LENGTH = 100000
import numpy as npy
import time 		# to make a runtime analysis
x = npy.array([1./(i+1) for i in range(ARRAY_LENGTH)])



direction = npy.zeros(ARRAY_LENGTH)
direction[0] = 1.

def run_test(f,x_0,direction, message='', print_derivatives=False):
	print message
	ax = npy.array([adouble(0.) for i in range(ARRAY_LENGTH)])

	start_time = time.time()
	trace_on(1)
	for i in range(ARRAY_LENGTH):
		ax[i].is_independent(x[i])# equivalent to ax[i]<<=x[i]
	ay = f(ax)
	y = depends_on(ay)
	trace_off()
	run_time_taping = time.time() - start_time

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
	
	print 'Adolc\tfunction taping:\t........\telapsed time: %f'%run_time_taping
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


#def f(avec):
	#a = avec[0]*avec[-1]
	#c =avec[0]*avec[0]
	#if isinstance(a,badouble):
		#print avec[0].location
		#print avec[1].location
		#print a.location
		#print c.location
	#return a
#y = run_test(f,x,direction)


#py_tape_doc(1,x,npy.array([y]))

#def f(avec):
	#return npy.sum(avec)
#run_test(f,x,direction)

def f(avec):
	return npy.prod(avec)
run_test(f,x,direction)

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
