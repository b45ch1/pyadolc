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
    * Debian Stretch Linux, Python 2.7.13, NumPy 1.13.1
    * OSX 10.9 (Mavericks), Python 2.7.6, NumPy 1.8.0
    * OSX 10.11 (El Capitan), Python 2.7.11, NumPy 1.10.11

BRANCHES:

    There are several branches available for different versions of ADOL-C/Colpack and Boost.
    In case you have issues installing the master branch, you can have a look there.


REQUIREMENTS:

    * C and C++ compiler
    * Python and Numpy, both with header files
    * ADOL-C, official versions from http://www.coin-or.org/projects/ADOL-C.xml
    * ColPack from http://cscapes.cs.purdue.edu/download/ColPack
    * boost::python Version 1.66 from http://www.boost.org/ or from the apt-get repository.

INSTALLATION UBUNTU / DEBIAN (Stretch):

    * install boost-python via apt-get
    * install autotools-dev libtool libboost-all-dev
    * Use ``./bootstrap.sh`` to download ADOL-C and ColPack and compile them.
    * Run ``python setup.py`` and follow the instructions

INSTALLATION OSX:

    * Run::

        brew install wget
        brew install automake
        brew install shtool
        brew install libtool
        brew install boost
        brew install boost-python
        brew link boost --force
        brew link boost-python --force

    * If you installed homebrew in the default location ``/usr/local``, you can skip this step.  Otherwise, if you installed homebrew somewhere else on your system, you will need to edit ``bootstrap.sh`` and ``setup.py``.  First, in the ColPack build section of ``bootstrap.sh``, add the flags::

        --with-boost-libdir='<homebrew_libdir>' --with-boost-includedir='<homebrew_includedir>'
      to the end of the ``./configure`` commands, where ``<homebrew_libdir>`` and ``<homebrew_includedir>`` are the locations of homebrew's ``lib`` and ``include`` directories, respectively.  Similarly, edit setup.py so that ``BOOST_DIR = '<homewbrew_root>'`` where ``<homebrew_root>`` is the base directory of your homebrew install (where ``lib``, ``include``, ... are located).

    * Run::

        ./bootstrap.sh
        CC=clang CXX=clang++ python setup.py build
        python setup.py install

   You may have to run ``brew link automake`` to generate symbolic links.


TEST YOUR INSTALLATION:

    * install nose, matplotlib, e.g., via pip install nose matplotlib
    * add pyadolc to your python path
    * run ``python -c "import adolc; adolc.test()"``.
      All tests should pass.
    * If anything goes wrong, please file a bug report.

    .. warning::

        If you run the test from the root folder of pyadolc you will get ``ImportError: No module named _adolc`` since it first looks in the local folder ``./adolc`` before trying the other directories in your PYTHONPATH.


MANUAL INSTALLATION:

    Follow the steps in ``./bootstrap.sh`` and adapt if necessary.
