#!/usr/bin/env python
import numpy as npy
from adolc import *
import time

def run_test(f,N,M, message='', print_derivatives=False):
	print message
	x = npy.array([1./(i+1) for i in range(N)])
	y = npy.zeros(M)
	ax = npy.array([adouble(0.) for i in range(N)])

	start_time = time.time()
	trace_on(1)
	for n in range(N):
		ax[n].is_independent(x[n])
	ay = f(ax)
	for m in range(M):
		y[m] = depends_on(ay[m])
	trace_off()
	tape_to_latex(1,x,y)
	runtime_taping = time.time() - start_time
	print 'Adolc\tfunction taping:\t........\telapsed time: %f'%runtime_taping

	start_time = time.time()
	y_adolc = function(1,x)
	runtime_adolc = time.time() - start_time
	print 'Adolc\tfunction evaluation:\t%f\telapsed time: %f'%(y_adolc[0],runtime_adolc)
	
	start_time = time.time()
	y_normal = f(x)
	runtime_normal = time.time() - start_time
	print 'normal\tfunction evaluation:\t%f\telapsed time: %f'%(y_normal[0],runtime_normal)

	if M==1:
		start_time = time.time()
		g = gradient(1,x)
		runtime_gradient = time.time() - start_time
		print 'gradient evaluation:\t\t........\telapsed time: %f'%runtime_gradient

	start_time = time.time()
	J = jacobian(1,x)
	runtime_jacobian = time.time() - start_time
	print 'jacobian evaluation:\t\t........\telapsed time: %f'%runtime_jacobian

	#start_time = time.time()
	#y_and_deriv = fos_forward(1,1,1,x,direction)
	#runtime_fos_forward = time.time() - start_time
	
	#if print_derivatives == True:
		 #print g
	#else:
		 #print ''
	#print 'direct. diff. evaluation:\t%f\telapsed time: %f'%(y_and_deriv['y'],runtime_fos_forward),
	#if print_derivatives == True:
		 #print  y_and_deriv['directional_derivative']
	#else:
		 #print ''
	#print ''
	#return y_normal


N = 500
M = 500
direction = npy.zeros(N)
direction[0] = 1.


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

#print "N=%d   M=%d"%(N,M)

#def f(avec):
	#return npy.array([npy.prod(avec)])
#run_test(f,N,1,'\n\nspeelpenning')


A = npy.zeros((M,N))
A[:] = [[ 1./N +(n==m) for n in range(N)] for m in range(M)]

def f(x):
	global A
	return npy.dot(A,x)
run_test(f,N,M,'\n\nmatrix vector multiplication')

