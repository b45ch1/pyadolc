from numpy.testing import *
import numpy
import numpy.random

from adolc import *
from adolc.cgraph import *
from adolc.tangent import *

class TangentOperationsTests(TestCase):
    def test_constructor(self):
        t1 = Tangent(1,2)
        t2 = Tangent(adouble(1),2)

    def test_float_tangent_float_tangent(self):
        tx = Tangent(2,3)
        ty = Tangent(5,7)
        
        tz = tx * ty
        assert_array_almost_equal([tz.x,tz.xdot], [2*5, 3*5 + 2*7])
        
        tz = tx + ty
        assert_array_almost_equal([tz.x,tz.xdot], [2+5, 3+7])
        
    def test_double_tangent_adouble(self):
        tx = Tangent(2,3)
        ay = adouble(5)
        
        tz = tx * ay
        assert_array_almost_equal([tz.x.val,tz.xdot.val], [2*5, 3*5])
        
        tz = tx + ay
        assert_array_almost_equal([tz.x.val,tz.xdot], [2+5, 3])        
        
        
    def test_adouble_tangent_adouble_addition(self):
        tx = Tangent(adouble(2), 1)
        ty = Tangent(adouble(3), 0)
        tz = tx + ty
        assert_array_almost_equal([tz.x.val,tz.xdot], [5, 1])



# class FunctionExampleTests(TestCase):
    # def test_utps_on_jacobian(self):
        
        # def f(x,p):
            # print p
            # print p[0] + p[1]
            # return (p[0] + p[1]) *  x**2
    
        # AP = AdolcProgram()
        # AP.trace_on(1)
        # ax = adouble(3.)
        # ap = adouble([5.,7.])
        # AP.independent(ax)
        # AP.independent(ap)
        
        # tp = [Tangent(ap[0],1),Tangent(ap[1],0)]
        # tf = f(ax,tp)
        
        # aJ = tf.xdot
        
        # print aJ
        
        # AP.dependent(aJ)
        # AP.trace_off()
        
        # g = gradient(1, [1,2,3])
        
        # print g
        
        

if __name__ == '__main__':
    try:
        import nose
    except:
        print 'Please install nose for unit testing'
    nose.runmodule()
