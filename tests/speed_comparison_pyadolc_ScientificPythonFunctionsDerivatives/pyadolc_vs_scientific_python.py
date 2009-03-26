from numpy import *
from Scientific.Functions.Derivatives import DerivVar
from  adolc import *
from time import time
N = 10
A = diag([i+1 for i in range(N)])

def f(x):
	return 0.5*dot(x,dot(A,x))

x = ones(N)
sx = array([DerivVar(1., i, 2) for i in range(N)])
ax = array([adouble(1.) for i in range(N)])

trace_on(0)
independent(ax)
y = f(ax)
dependent(y)
trace_off()



start_time = time()
adolc_H = hessian(0,x)
end_time = time()
adolc_runtime = (end_time-start_time)
print 'adolc: elapsed time = %f sec' %adolc_runtime

start_time = time()
scientific_H = array(f(sx)[2])
end_time = time()
scientific_runtime = (end_time-start_time)
print 'Scientific: elapsed time = %f sec' %scientific_runtime

print 'ratio time  adolc/Scientific Python:  %f'%(adolc_runtime/scientific_runtime)
assert prod(scientific_H == adolc_H)






