import distutils.sysconfig
import numpy

LIBS		= ['adolc']
LIBPATH		= ['/data/walter/opt_software/adolc-1.10.2/lib']
INCLUDEPATH	= ['/data/walter/opt_software/adolc-1.10.2/include/adolc']

env = Environment(
	CPPPATH=[distutils.sysconfig.get_python_inc(),numpy.get_include()] + INCLUDEPATH,
	CXXFLAGS="-ftemplate-depth-100 -DBOOST_PYTHON_DYNAMIC_LIB -O2",
	LIBPATH=["/usr/lib/python2.5/config"] + LIBPATH,
	LIBS=["boost_python"] + LIBS,
	SHLIBPREFIX="", #gets rid of lib prefix
)
Default('.')
env.SharedLibrary(target='Adolc', source=['py_adolc.cpp', 'num_util.cpp'])

