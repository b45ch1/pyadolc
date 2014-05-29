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
        tx = Tangent(2.,3.)
        ty = Tangent(5.,7.)
        
        tz = tx + ty
        assert_array_almost_equal([tz.x,tz.xdot], [2+5, 3+7])
        
        tz = tx - ty
        assert_array_almost_equal([tz.x,tz.xdot], [2-5, 3-7])

        tz = tx * ty
        assert_array_almost_equal([tz.x,tz.xdot], [2*5, 3*5 + 2*7])
        
        tz = tx / ty
        assert_array_almost_equal([tz.x,tz.xdot], [2./5., (3*5 - 2*7.)/5**2])
        
    def test_double_tangent_adouble(self):
        tx = Tangent(2,3)
        ay = adouble(5)
        
        tz = tx + ay
        assert_array_almost_equal([tz.x.val,tz.xdot], [2+5, 3])           
        
        tz = tx * ay
        assert_array_almost_equal([tz.x.val,tz.xdot.val], [2*5, 3*5])
        
    def test_adouble_tangent_adouble_addition(self):
        tx = Tangent(adouble(2), 1)
        ty = Tangent(adouble(3), 0)
        tz = tx + ty
        assert_array_almost_equal([tz.x.val,tz.xdot], [5, 1])


class SemiImplicitOdeLhsTest(TestCase):
    """
    This is a test example taken from PYSOLVIND
    
    In chemical engineering, semi-implicit ODEs of the type::
    
        d/dt g(t,y(t)) = f(t,y(t))
        y(0) = y_0
    
    have to be solved. PYSOLVIND requires a function afcn that computes::
    
        d/dy g(t,y) d/dt y
        
        where d/dt y = xdd
                   y = xd
    """
    
    def test_differentiation_of_gfcn(self):
        def gfcn(a):
            print('called gfcn')
            ty = [Tangent(a.xd[0], a.xdd[0]),Tangent(a.xd[1], a.xdd[1]), Tangent(a.xd[2], a.xdd[2])]
            tlhs = [ty[0] * ty[2], ty[1] * ty[2], ty[2]]
            
            a.lhs[0] = tlhs[0].xdot
            a.lhs[1] = tlhs[1].xdot
            a.lhs[2] = tlhs[2].xdot
        
        def afcn(a):
            a.lhs[0] = a.xd[2] * a.xdd[0] + a.xd[0] * a.xdd[2]
            a.lhs[1] = a.xd[2] * a.xdd[1] + a.xd[1] * a.xdd[2]
            a.lhs[2] = a.xdd[2]        

        class Args:
            def __init__(self):
                self.xd  = numpy.random.rand(3)
                self.xdd = numpy.random.rand(3)
                self.lhs = numpy.zeros(3)
                
        args = Args()
        
        gfcn(args)
        result1 = args.lhs.copy()
        
        afcn(args)
        result2 = args.lhs.copy()
        
        assert_array_almost_equal(result1, result2)

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
        print('Please install nose for unit testing')
    nose.runmodule()
