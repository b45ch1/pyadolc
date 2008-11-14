import distutils.sysconfig
import numpy

LIBS		= ['adolc',
				#'boost_python-gcc42-mt-1_34_1'
				'boost_python-gcc42-1_34_1'
			]
LIBPATH		= [
				'/u/walter/workspace/python_extension_for_adolc/adolc-1.11.0-trunk/lib',
				#'/u/walter/workspace/python_extension_for_adolc/adolc-1.10.2/lib',
				#'/data/walter/opt_software/boost_1_34_1/bin.v2/libs/python/build/gcc-4.2.1/release/threading-multi'
				'/data/walter/opt_software/boost_1_34_1/bin.v2/libs/python/build/gcc-4.2.1/release'
		  ]
INCLUDEPATH	= [
			'/u/walter/workspace/python_extension_for_adolc/adolc-1.11.0-trunk/include',
			'/u/walter/opt_software/my_global_cpp_libaries',
			#'/u/walter/workspace/python_extension_for_adolc/adolc-1.10.2/include/adolc',
			'/data/walter/opt_software/boost_1_34_1',
			'/usr/include/python2.5'
			]

env = Environment(
	CPPPATH=[distutils.sysconfig.get_python_inc(),numpy.get_include()] + INCLUDEPATH,
	CXXFLAGS="-ftemplate-depth-100 -DBOOST_PYTHON_DYNAMIC_LIB -O2",
	LIBPATH=["/usr/lib/python2.5/config"] + LIBPATH,
	LIBS= LIBS,
	RPATH = LIBPATH, #include information where shared libraries can be found to avoid errors like: "ImportError: libboost_python-gcc42-mt-1_34_1.so.1.34.1: cannot open shared object file: No such file or directory"
	SHLIBPREFIX="", #gets rid of lib prefix
)
Default('.')
adolc = env.SharedLibrary(target='adolc', source=['py_adolc.cpp', 'num_util.cpp'])
env.Install("./release/adolc", adolc)

