import adolc
import numpy

M,N = 4,2

sparsity_pattern_list = [numpy.random.randint(0,4*N,M)//(3*N) for n in range(N)]


def F(x):
    y = numpy.ones(M, dtype=x.dtype)
    for n,sp in enumerate(sparsity_pattern_list):
        for ns, s in enumerate(sp):
            if s == 1:
                y[ns] *= x[n]
        
    return y
        
x = numpy.random.rand(N)

adolc.trace_on(0)
x = adolc.adouble(x)
adolc.independent(x)
y = F(x)
adolc.dependent(y)
adolc.trace_off()

x = numpy.random.rand(N)
y = F(x)
y2 = adolc.function(0,x)
assert numpy.allclose(y,y2)

options = numpy.array([0,0,0,0],dtype=int)
pat = adolc.sparse.jac_pat(0,x,options)
result = adolc.colpack.sparse_jac_no_repeat(0,x,options)

print(adolc.jacobian(0,x))
print(pat)

print(result)


