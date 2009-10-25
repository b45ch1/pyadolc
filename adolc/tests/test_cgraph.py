from numpy.testing import *
import numpy
import numpy.random

from adolc import *
from adolc.cgraph import *

class TestAdolcProgram(TestCase):
    def test_forward(self):
        x = numpy.random.rand(*(2,3,4))
        y = numpy.random.rand(*(4,3,2))
        AP = AdolcProgram()
        AP.trace_on(1)
        ax = adouble(x)
        ay = adouble(y)
        AP.independent(ax)
        AP.independent(ay)
        az = ay.T * ax / ay.T
        AP.dependent(az)
        AP.trace_off()
        
        D,P = 3,4
        Vx = numpy.ones( x.shape + (P,D))
        Vy = numpy.ones( y.shape + (P,D))
        
        z = AP.forward([x,y])
        assert_array_almost_equal(z[0],x)
        
        z,W = AP.forward([x,y],[Vx,Vy],keep=False)
        assert_array_almost_equal(z[0],x)
        assert_array_almost_equal(W[0],Vx)
        
        D,P = 3,1
        Vx = numpy.ones( x.shape + (P,D))
        Vy = numpy.ones( y.shape + (P,D))
        z,W = AP.forward([x,y],[Vx,Vy],keep=D+1)
        assert_array_almost_equal(z[0],x)
        assert_array_almost_equal(W[0],Vx)

    def test_forward_scalar_independent_variables(self):
        x = 1.
        y = 2.
        AP = AdolcProgram()
        AP.trace_on(1)
        ax = adouble(x)
        ay = adouble(y)
        AP.independent(ax)
        AP.independent(ay)
        az = ax * ay
        AP.dependent(az)
        AP.trace_off()
        
        P,D = 3,5
        x = [1.]
        y = [2.]        
        Vx = numpy.ones((1,P,D))
        Vy = numpy.ones((1,P,D))
        
        AP.forward([x,y],[Vx,Vy])

        
    def test_reverse(self):
        x = numpy.random.rand(*(2,3,4))
        y = numpy.random.rand(*(4,3,2))
        AP = AdolcProgram()
        AP.trace_on(1)
        ax = adouble(x)
        ay = adouble(y)
        AP.independent(ax)
        AP.independent(ay)
        az = ay.T * ax / ay.T
        AP.dependent(az)
        AP.trace_off()
        
        D,P = 2,4
        Vx = numpy.ones( x.shape + (P,D))
        Vy = numpy.ones( y.shape + (P,D))
        z,W = AP.forward([x,y],[Vx,Vy])
        Q = 7
        Wbar = numpy.random.rand( *( (Q,) + z[0].shape + (P,D+1,)))
        Vbar_list = AP.reverse([Wbar])
        
        assert_array_equal(Vbar_list[0].shape,(Q,2,3,4,P,D+1))
        assert_array_almost_equal(Vbar_list[0],Wbar)
        
        
    def test_jacobian(self):
        A = numpy.random.rand(5,4)
        x = numpy.random.rand(4)
        y = numpy.random.rand(4)
        
        def f(x):
            return numpy.dot(A,x)
        
        AP = AdolcProgram()
        AP.trace_on(1)
        ax = adouble(x)
        AP.independent(ax)
        af = f(ax) 
        AP.dependent(af)
        AP.trace_off()
        
        J = AP.jacobian([y])
        assert_array_almost_equal(J, A)
        
        
        
if __name__ == "__main__":
    run_module_suite()

