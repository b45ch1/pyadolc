=============================
PYADOLC, a wrapper for ADOL-C
=============================

Short Description:
    This PYADOLC, a Python module to differentiate complex algorithms written in Python.
    It wraps the functionality of the library ADOL-C (C++).

Author:
    Sebastian F. Walter

Licence (new BSD):
    Copyright (c) 2008, Sebastian F. Walter
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the HU Berlin nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Sebastian F. Walter ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Sebastian F. Walter BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


EXAMPLE USAGE::

    import numpy
    from adolc import *
    N = M = 10
    A = numpy.zeros((M,N))
    A[:] = [[ 1./N +(n==m) for n in range(N)] for m in range(M)]
    def f(x):
        return numpy.dot(A,x)

    # tape a function evaluation
    ax = numpy.array([adouble(0) for n in range(N)])
    trace_on(1)
    independent(ax)
    ay = f(ax)
    dependent(ay)
    trace_off()

    x = numpy.array([n+1 for n in range(N)])

    # compute jacobian of f at x
    J = jacobian(1,x)

    # compute gradient of f at x
    if M==1:
        g = gradient(1,x)


REQUIREMENTS:
    * Python and Numpy, both with header files
    * ADOL-C version >=2.3.0  http://www.coin-or.org/projects/ADOL-C.xml
    * boost::python from http://www.boost.org/
    * scons build tool (makes things easier if you need to recompile pyadolc)


OPTIONAL REQUIREMENTS:
    * For sparse Jacobians and Hessians: ColPack >=1.0.6 http://www.cscapes.org/coloringpage/software.htm


KNOWN TO WORK WITH
    * Known to work for Ubuntu Linux, Python 2.6, NumPy 1.3.0, Boost:Python 1.40.0, ADOL-C 2.3.0, ColPack 1.0.6
    * Known to work for Ubuntu Linux, Python 2.7.3, NumPy 1.8.0, Boost:Python 1.48.0, ADOL-C 2.6.0, ColPack 1.0.9


INSTALLATION:

    * CHECK REQUIREMENTS: Make sure you have ADOL-C (version 2.6.0), ColPack (version 1.0.9) the boost libraries and numpy installed. All with header files.
    * BUILD COLPACK
        * if you have 32bit system: run ``./configure --prefix=~/workspace/adol-c/ThirdParty/ColPack/``
        * if you have 64bit system: run ``./configure --prefix=~/workspace/adol-c/ThirdParty/ColPack/ --libdir='${prefix}/lib64'``
        * run ``make && make install``
        * this should generate ``~/workspace/adol-c/ThirdParty/ColPack/lib64/libColPack.so``.
    * BUILD ADOL-C:
        * run ``./configure --enable-sparse --with-colpack=~/workspace/adol-c/ThirdParty/ColPack/``
        * REMARK: the option ``--enable-sparse`` is used in ADOLC-2.2.1. In ADOLC-2.1.0 it is called ``--with-sparse``.
        * run ``make``
        * You don't have to run ``make install``.
        * You should then have a folder ``~/workspace/ADOL-C-2.1.0/ADOL-C`` with  ``adolc/adolc.h`` in it.
    * CLONE PYADOLC: ``cd ~/workspace/adol-c`` and then ``git clone git://github.com/b45ch1/pyadolc.git python``
      You should then have a folder ~/workspace/adol-c/python containing the file SConstruct
    * if you get no permission errors, then use the https url from github to clone the repository.
    * BUILD PYADOLC:
        Go to the folder ~/workspace/adol-c/python and run ``scons``
        This should compile and link everything you need.
    * TEST YOUR INSTALLATION:
        * run ``python -c "import adolc; adolc.test()"``. All tests should pass.
    * If anything goes wrong, please file a bug report.

