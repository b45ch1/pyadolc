import distutils.sysconfig
import numpy

LIBS		= ['adolc','libboost_python-gcc42-mt-1_34_1.so.1.34.1']
LIBPATH		= [	'/data/walter/opt_software/adolc-1.10.2/lib',
			'/data/walter/opt_software/boost_1_34_1/bin.v2/libs/python/build/gcc-4.2.1/release/threading-multi'
		  ]
INCLUDEPATH	= ['/data/walter/opt_software/adolc-1.10.2/include/adolc','/data/walter/opt_software/boost_1_34_1']

env = Environment(
	CPPPATH=[distutils.sysconfig.get_python_inc(),numpy.get_include()] + INCLUDEPATH,
	CXXFLAGS="-ftemplate-depth-100 -DBOOST_PYTHON_DYNAMIC_LIB -O2",
	LIBPATH=["/usr/lib/python2.5/config"] + LIBPATH,
	LIBS= LIBS,
	SHLIBPREFIX="", #gets rid of lib prefix
)
Default('.')
env.SharedLibrary(target='Adolc', source=['py_adolc.cpp', 'num_util.cpp'])

