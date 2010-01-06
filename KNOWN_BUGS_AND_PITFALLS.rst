Rationale:

    This text collects common pitfalls and known bugs that may lead to incorrect computation of derivatives.


Pitfalls
========

    1. Inplace Numpy Array Operations

        One has to be caerful with the operations `+=` , `*=`, ... of numpy arrays.
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

        This is not a bug of PYADOLC but the way `*=` is implemented in numpy. One can see similar behaviour for the dtype int and float::

            In [1]: import numpy

            In [2]: x = numpy.ones(2,dtype=int)

            In [3]: y = 1.3 * numpy.ones(2,dtype=float)

            In [4]: z = x * y

            In [5]: z
            Out[5]: array([ 1.3,  1.3])

            In [6]: x *= y

            In [7]: x
            Out[7]: array([1, 1])

            In [8]: x.dtype
            Out[8]: dtype('int32')

        that means that the inplace operation `x *= y ` is *not* the same as `x = x * y`.
        This also means that::

            x = adouble(numpy.array([1,2,3],dtype=float))
            
        It is inconsistent to the Python behaviour and therefore a little surprising::

            In [9]: a = 1

            In [10]: b = 1.3

            In [11]: c = a * b

            In [12]: c
            Out[12]: 1.3

            In [13]: a *= b

            In [14]: a
            Out[14]: 1.3

        This is intended behaviour of numpy, but it leads to incorrect computations since no exception or
        warning is raised by numpy. For more info see
        http://www.mail-archive.com/numpy-discussion@lists.sourceforge.net/msg03236.html

    