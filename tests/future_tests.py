"""
This file is to check if future API changes of Python/Numpy/...
will still work with pyadolc.

"""

from __future__ import division
from unit_test import *


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
