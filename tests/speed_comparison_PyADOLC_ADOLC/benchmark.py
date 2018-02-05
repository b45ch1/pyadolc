#!/usr/bin/env python
import numpy
from adolc import *
import time

def speelpenning(x):
	return numpy.array([numpy.prod(x)])

def matrix_vector_multiplication(x):
	global A
	return numpy.dot(A,x)


def benchmark(f,N,M,message):
	print(message)
	x = numpy.array([1./(i+1) for i in range(N)])
	y = numpy.zeros(M)
	ax = adouble(x)
	
	start_time = time.time()
	trace_on(1)
	independent(ax)
	ay = f(ax)
	dependent(ay)
	trace_off()
	#tape_to_latex(1,x,y)

	print('N=%d,M=%d'%(N,M))
	
	runtime_taping = time.time() - start_time
	print('PyADOLC\tfunction taping:\t........\telapsed time: %f'%runtime_taping)

	start_time = time.time()
	y_adolc = function(1,x)
	runtime_adolc = time.time() - start_time
	print('Adolc\tfunction evaluation:\t%f\telapsed time: %f'%(y_adolc[0],runtime_adolc))
	
	start_time = time.time()
	y_normal = f(x)
	runtime_normal = time.time() - start_time
	print('normal\tfunction evaluation:\t%f\telapsed time: %f'%(y_normal[0],runtime_normal))

	start_time = time.time()
	J = jacobian(1,x)
	runtime_jacobian = time.time() - start_time
	print('jacobian evaluation:\t\t........\telapsed time: %f'%runtime_jacobian)

	if M==1:
		start_time = time.time()
		g = gradient(1,x)
		runtime_gradient = time.time() - start_time
		print('gradient evaluation:\t\t........\telapsed time: %f'%runtime_gradient)

if __name__ == "__main__":
	import sys
	if len(sys.argv)==3:
		N = int(sys.argv[1])
		M = int(sys.argv[2])
	else:
		N = 10
		M = 1

	benchmark(speelpenning,N,1,"\n\nspeelpenning:\n")

	A = numpy.zeros((M,N))
	A[:] = [[ 1./N +(n==m) for n in range(N)] for m in range(M)]
	benchmark(matrix_vector_multiplication,N,M,"\n\nmatrix vector multiplication:\n")
	

