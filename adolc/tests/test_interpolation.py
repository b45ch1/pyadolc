from numpy.testing import *
import numpy
import numpy.random

from adolc import *
from adolc.linalg import *
from adolc.interpolation import *


class InterpolationTests(TestCase):

    def test_entangle(self):
        # setup inputs for entangle_cross

        V1  = numpy.random.random((4,2))
        V2  = numpy.random.random((4,2))
        V12 = numpy.random.random((4,2,2))
        V   = numpy.zeros((4,8,2))


        # execute entangle_cross

        entangle_cross(V, V1, V2, V12)


        # setup inputs for detangle_cross

        U1  = numpy.zeros((4,2))
        U2  = numpy.zeros((4,2))
        U12 = numpy.zeros((4,2,2))


        # execute detangle_cross

        detangle_cross(V, U1, U2, U12)


        # check correctness

        assert_array_almost_equal(U1,  V1)
        assert_array_almost_equal(U2,  V2)
        assert_array_almost_equal(U12, V12)


if __name__ == '__main__':
    try:
        import nose
    except:
        print('Please install nose for unit testing')
    nose.runmodule()
