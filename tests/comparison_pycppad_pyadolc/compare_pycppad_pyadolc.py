import time
import timeit
import pylab
import numpy
import numpy.random
import pycppad 
import adolc 


if __name__ == "__main__":
	N_max = 120
	reps = 100

	Ns = list(range(2,N_max,5))
	adolc_gradient_runtimes = []
	cppad_gradient_runtimes = []
	adolc_hessian_runtimes = []
	cppad_hessian_runtimes = []
	
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
		
		cppad_hessian_runtime  = timeit.Timer('f.hessian(x, w)', 'from __main__ import f,x,w').timeit(number=reps)/reps
		cppad_gradient_runtime = timeit.Timer('f.jacobian(x)', 'from __main__ import f,x').timeit(number=reps)/reps
		
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
		adolc_hessian_runtime = timeit.Timer('adolc.hessian(0, x)', 'import adolc;from __main__ import x').timeit(number=reps)/reps
		adolc_gradient_runtime = timeit.Timer('adolc.gradient(0, x)', 'import adolc;from __main__ import x').timeit(number=reps)/reps
		
		adolc_hessian_runtimes.append(adolc_hessian_runtime)
		cppad_hessian_runtimes.append(cppad_hessian_runtime)
		
		adolc_gradient_runtimes.append(adolc_gradient_runtime)
		cppad_gradient_runtimes.append(cppad_gradient_runtime)
	
	pylab.figure()
	pylab.semilogy(Ns, adolc_hessian_runtimes, '-b.', label='hessian pyadolc')
	pylab.semilogy(Ns, adolc_gradient_runtimes, '-bd', label='gradient pyadolc')\
	
	pylab.semilogy(Ns, cppad_hessian_runtimes, '-r.', label='hessian pycppad')
	pylab.semilogy(Ns, cppad_gradient_runtimes, '-rd', label='gradient pycppad')
	
	
	pylab.xlabel(r'problem size $N$')
	pylab.ylabel(r'average runtime $t= \frac{1}{%d} \sum_{r=1}^{%d} t_r$  in [sec]'%(reps,reps))
	pylab.title(r'Hessian/Gradient computation, %d runs '%reps)
	pylab.legend(loc=2)
	pylab.savefig('runtimes_pycppad_pyadolc.png')
	pylab.show()
