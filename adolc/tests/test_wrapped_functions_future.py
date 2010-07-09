"""
In this file, there should be tests that check  if future API changes of Python/Numpy/...
will break pyadolc.
"""

from __future__ import division
from numpy.testing import *

from adolc import *


def test_truediv2():
    x=1
    y=2
    ax=adouble(x)
    ay=adouble(y)

    z= x.__truediv__(y)
    az1 = ax/y
    az2 = x/ay
    az3 = ax/ay

    assert_almost_equal(az1.val, z)
    assert_almost_equal(az2.val, z)
    assert_almost_equal(az3.val, z)


if __name__ == '__main__':
    try:
        import nose
    except:
        print 'Please install nose for unit testing'
    nose.runmodule()