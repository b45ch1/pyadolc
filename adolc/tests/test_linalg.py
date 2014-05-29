from numpy.testing import *
import numpy
import numpy.random

from adolc import *
from adolc.linalg import *


class LinearAlgebraTests(TestCase):
    def test_qr_decomposition(self):
       N = 2
       
       trace_on(0)
       aA = numpy.array([[adouble(numpy.random.rand()) for c in range(N)] for r in range(N)])
       independent(aA)
       aQ,aR =  qr(aA)
       aB = numpy.dot(aQ,aR)
       dependent(aQ)
       dependent(aR)
       dependent(aB)
       trace_off()
       
       P,D = 3,5
       A  = numpy.random.rand(N,N)
       VA = numpy.random.rand(N,N,P,D)
       
       rA  = A.reshape(N**2)
       rVA = VA.reshape((N**2, P,D))
       rQRB, rVQRB = hov_forward(0, rA, rVA)
       
       rQ = rQRB[:N**2]
       rR = rQRB[N**2:2*N**2]
       rB = rQRB[2*N**2:]
       
       rVQ = rVQRB[:N**2,...]
       rVR = rVQRB[N**2:2*N**2,...]
       rVB = rVQRB[2*N**2:,...]
       
       Q = rQ.reshape((N,N))
       R = rR.reshape((N,N))
       B = rB.reshape((N,N))
       
       VQ = rVQ.reshape((N,N,P,D))
       VR = rVR.reshape((N,N,P,D))
       VB = rVB.reshape((N,N,P,D))

       assert_array_almost_equal(B,A)
       assert_array_almost_equal(VB,VA)


if __name__ == '__main__':
    try:
        import nose
    except:
        print('Please install nose for unit testing')
    nose.runmodule()
