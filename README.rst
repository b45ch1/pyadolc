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


THIS VERSION OF PYADOLC IS KNOWN TO WORK WITH:

    * Ubuntu Linux, Python 2.7.3, NumPy 1.8.0
    * OSX 10.9 (Mavericks), Python 2.7.6, NumPy 1.8.0
    * OSX 10.11 (El Capitan), Python 2.7.11, NumPy 1.10.11


REQUIREMENTS:
    * gcc
    * Python and Numpy, both with header files
    * ADOL-C, official versions from http://www.coin-or.org/projects/ADOL-C.xml 
    * boost::python from http://www.boost.org/

INSTALLATION UBUNTU:

    * install boost-python via apt-get
    * Use ``./bootstrap.sh`` to download ADOL-C and ColPack and compile them.
    * Run ``python setup.py``

INSTALLATION OSX:

    * Run::

        brew install wget
        brew install automake
        brew install shtool
        brew install libtool
        brew install boost --with-python
        brew install boost-python
        brew install homebrew/science/adol-c

    * Run ``CC=clang CXX=clang++ python setup.py``

   You may have to run``brew link automake`` to generate symbolic links.


TEST YOUR INSTALLATION:

    * install nose, e.g., via pip install nose
    * run ``python -c "import adolc; adolc.test()"``.
      All tests should pass.
    * If anything goes wrong, please file a bug report.

MANUAL INSTALLATION:

    Follow the steps in ``./bootstrap.sh`` and adapt if necessary.
