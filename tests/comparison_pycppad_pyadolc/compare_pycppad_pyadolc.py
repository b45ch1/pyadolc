import time
import timeit
import pylab
import numpy
import numpy.random
import pycppad 
import adolc 


if __name__ == "__main__":
	N_max = 200
	reps = 30

	Ns = range(2,N_max,5)
	adolc_runtimes = []
	cppad_runtimes = []
	
	for N in Ns:
	
		# cppad timing
		x     = numpy.zeros(N, dtype=float)
		ax   = pycppad.independent(x)
		
		atmp = []
		for n in range(N):
			atmp.append(numpy.sin( numpy.sum(ax[:n])))
		ay   = numpy.array( [ ax[0] * numpy.sin( numpy.sum(atmp)) ] )
		f   = pycppad.adfun(ax, ay)
		x   = numpy.random.rand(N)
		w   = numpy.array( [ 1.] ) # compute Hessian of x0 * sin(x1)
		
		cppad_runtime = timeit.Timer('f.hessian(x, w)', 'from __main__ import f,x,w').timeit(number=reps)/reps
		
		
		# adolc timing
		x     = numpy.zeros(N, dtype=float)
		adolc.trace_on(0)
		ax = adolc.adouble(x)
		adolc.independent(ax)
		atmp = []
		for n in range(N):
			atmp.append(numpy.sin( numpy.sum(ax[:n])))
		ay   = numpy.array( [ ax[0] * numpy.sin( numpy.sum(atmp)) ] )
		adolc.dependent(ay)
		adolc.trace_off()
		
		x   = numpy.random.rand(N)
		adolc_runtime = timeit.Timer('adolc.hessian(0, x)', 'import adolc;from __main__ import x').timeit(number=reps)/reps

		adolc_runtimes.append(adolc_runtime)
		cppad_runtimes.append(cppad_runtime)
	
	pylab.figure()
	pylab.semilogy(Ns, adolc_runtimes, '-b.', label='pyadolc')
	pylab.semilogy(Ns, cppad_runtimes, '-r.', label='pycppad')
	pylab.xlabel(r'problem size $N$')
	pylab.ylabel(r'runtime t [sec]')
	pylab.title(r'Hessian computation')
	pylab.legend()
	pylab.savefig('runtimes_pycppad_pyadolc.png')
	pylab.show()
