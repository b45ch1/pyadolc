from numpy.testing import *
import numpy
import numpy.random

from adolc.finitedifferences import *




class FiniteDifferencesTest(TestCase):
    def test_jacobian(self):
        N = 10
        A = numpy.random.rand(N)
        
        def f(x):
            return numpy.dot(A, x)
        
        x = numpy.random.rand(N)
        
        assert_array_almost_equal(jacobian(f,x),A, decimal=7)
        
    
    
    
    
    
if __name__ == '__main__':
    try:
        import nose
    except:
        print('Please install nose for unit testing')
    nose.runmodule()
