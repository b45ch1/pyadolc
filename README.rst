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


EXAMPLE USAGE:
--------------

>>> import numpy
>>> from adolc import *
>>> N = M = 10
>>> A = numpy.zeros((M,N))
>>> A[:] = [[ 1./N +(n==m) for n in range(N)] for m in range(M)]
>>> def f(x):
>>>     return numpy.dot(A,x)
>>>
>>> # tape a function evaluation
>>> ax = numpy.array([adouble(0) for n in range(N)])
>>> trace_on(1)
>>> independent(ax)
>>> ay = f(ax)
>>> dependent(ay)
>>> trace_off()
>>> 
>>> x = numpy.array([n+1 for n in range(N)])
>>> 
>>> # compute jacobian of f at x
>>> J = jacobian(1,x)
>>> 
>>> # compute gradient of f at x
>>> if M==1:
>>> 	g = gradient(1,x)


REQUIREMENTS:
	* Python and Numpy, both with header files
	* boost::python from http://www.boost.org/


INSTALLATION:
    1) copy this folder to a place where it is going to stay.
       This is imporant since the path to the  shared library ``adolc.so`` of adolc  is saved in the adolc.so file for python as absolute path (i.e. the ``RPATH`` is set).
    2) go to the folderadolc-2.0.0 and compile ADOL-C:
       ``./configure && make``	do *NOT* ``make install``
    3) rename ``setup.py.EXAMPLE`` to ``setup.py`` to fit your system:
       Run  ``python setup.py build_ext --inplace``.
       Alternatively you can rename ``SConstruct.EXAMPLE`` to ``SConstruct`` and modify it to fit your system and build with ``scons``.
       Using ``scons`` is more convenient to work with when you often pull new versions:
       when using the Python distutils you have to remove the oldbinaries before you can run
       ``python setup.py build_ext --inplace again``. In the development process it is likely that the ``scons`` version works whereas ``setup.py`` fails because of new features in the code.
    4) Add the directory to your ``PYTHONPATH``.
       E.g. add the following line in your ``~/.bashrc`` file:
       ``export PYTHONPATH=$PYTHONPATH:/home/walter/workspace/pyadolc``


INSTALLATION OF SPARSE SUPPORT:
	To use the functionality of sparse jacobians you will need Colpack (http://www.cs.odu.edu/~dnguyen/dox/colpack/html/).
	For convenience I have already improved the makefile of ColPack and uploaded it to
	http://github.com/b45ch1/colpack/tree/master
	
	1) go to ``./colpack``
	2) ``make``     to compile
	3) ``make install`` to copy header files and shared library to ./colpack/build/include and ./colpack/build/lib
	4) copy the contents of ``./colpack/build/include`` and ``./colpack/build/lib``  to ``./pyadolc/adolc-2.0.0/colpack``
	5) run ``scons``


