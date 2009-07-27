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
    * ADOL-C version 2.1.0 http://www.coin-or.org/projects/ADOL-C.xml
    * boost::python from http://www.boost.org/
    * scons build tool

OPTIONAL REQUIREMENTS:
    * Fork of Colpack http://github.com/b45ch1/colpack . Colpack is needed for sparse Jacobians and sparse Hessians. The original version does not work for me...


INSTALLATION:
Warning 1:
this version of PYADOLC only works with ADOL-C Version 2.1.0

Warning 2:
At the moment, the installation using setup.py is lagging behind. Below, the way I would install ``PYADOLC``. If you do it another way, send me an email or fork this project and I'll incorporate it. 
Alternatively you can _try_ to use distutils, but support for it lags behind scons: i.e. 1) rename ``setup.py.EXAMPLE`` to ``setup.py`` to fit your system, then 2) run  ``python setup.py build_ext --inplace``.


INSTALLATION:

    * Build ADOL-C: run ``./configure && make``. To use PYADOLC with sparse support, you do _not_ have to do ``./configure --with-sparse``. You should then have a folder ``~/workspace/ADOL-C-2.1.0/ADOL-C/src`` with  ``adolc.h`` in it.
    * ``cd ~`` and then ``git clone git://github.com/b45ch1/pyadolc.git``
    *  Rename ``SConstruct.EXAMPLE`` to ``SConstruct`` and modify it to fit your system. In your example, ``adolc_source_dir`` in the SConstruct file should be ``~/workspace/ADOL-C-2.1.0/ADOL-C/src``
    *  Run ``~/pyadolc/scons``, this will create the shared libraries ``_adolc.so`` and ``_sparse.so``. If you didn't do the (OPTIONAL) steps below, it will only compile ``_adolc.so`` but fail to compile ``_sparse.so``, i.e. you will get some compiler errors. However, everything but the sparse functionality will work!
    * (OPTIONAL) Download colpack with ``git clone git://github.com/b45ch1/colpack.git  `` and run ``make``. You should then have a folder ``~/colpack/build`` with subfolders ``lib`` and ``include``
    * (OPTIONAL) Copy everything in ``~/colpack/build/include`` to ``~/workspace/ADOL-C-2.1.0/ThirdParty/ColPack``
    * (OPTIONAL) Run ``~/pyadolc/scons`` and check that ``_sparse.so`` has compiled.
    * Add the directory to your ``PYTHONPATH``. E.g. add the following line in your ``~/.bashrc`` file: ``export PYTHONPATH=$PYTHONPATH:/home/walter/workspace/pyadolc``
    * (OPTIONAL) Unit test: To check everything works you can run ``~/pyadolc/py.test tests/unit_test.py``.  In Debian it is in the python-codespeak-lib package and can be installed with ``apt-get install python-codespeak-libs``.
