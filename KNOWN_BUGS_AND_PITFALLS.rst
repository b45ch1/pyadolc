Rationale:

    This text collects common pitfalls and known bugs that may lead to incorrect computation of derivatives.


Pitfalls
========

    1. Inplace Numpy Array Operations

        One has to be careful with the operations `+=` , `*=`, ... of numpy arrays.
        The problem is best explained at an example::

            In [1]: import numpy

            In [2]: import adolc

            In [3]: x = adolc.adouble(numpy.array([1,2,3],dtype=float))

            In [4]: y = numpy.array([4,5,6],dtype=float)

            In [5]: x
            Out[5]: array([1(a), 2(a), 3(a)], dtype=object)

            In [6]: y
            Out[6]: array([ 4.,  5.,  6.])

            In [7]: x * y
            Out[7]: array([4(a), 10(a), 18(a)], dtype=object)

            In [8]: y *= x

            In [9]: y

            Out[9]: array([ 4.,  5.,  6.])

        That means that the inplace operation `x *= y ` is *not* the same as `x = x * y`.

        This is not a bug of PYADOLC but a design choice in numpy's implementation of the augmented
        assignment statements `*=`, etc. for arrays of objects.

        Numpy tries to cast the dtype of `y` to the dtype `x`. If x has dtype `float` then on each element
        y[i].__float__() is called.
