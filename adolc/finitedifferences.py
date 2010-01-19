
import numpy


def jacobian(ffcn, x, epsilon = 10**-8):
    """
    Computes the Jacobian of the function ffcn by finite differences in point x.
    """
    
    N = numpy.size(x)
    xv = numpy.zeros((N,N))
    V = epsilon * numpy.eye(N)
    
    for n in range(N):
        xv[:,n] = x
    
    J = (ffcn(xv + V) - ffcn(xv))/epsilon
    return J
    
