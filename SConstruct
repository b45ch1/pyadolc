import distutils.sysconfig
import os
import numpy
import inspect

# -1: set environment variables to match your system
#     e.g. export COLPACK_DIR=/path/to/colpack
#     where /path/to/colpack contains the folders ./include and ./lib64
#     which contain libColPack.so and the include files
BASEDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

ADOLC_DIR   = os.environ.get('ADOLC_DIR', os.path.join(BASEDIR, 'PACKAGES/ADOL-C'))
COLPACK_DIR = os.environ.get('COLPACK_DIR', os.path.join(BASEDIR, 'PACKAGES/ADOL-C/ThirdParty/ColPack'))

adolc_include_path   = os.path.join(ADOLC_DIR, 'ADOL-C/include')
adolc_library_path   = os.path.join(ADOLC_DIR, 'ADOL-C/.libs')

colpack_include_path = os.path.join(COLPACK_DIR, 'include')
colpack_lib_path1    = os.path.join(COLPACK_DIR, 'lib')
colpack_lib_path2    = os.path.join(COLPACK_DIR, 'lib64')


LIBS        = ['adolc',
                'boost_python',
                'ColPack',
            ]
LIBPATH     = [
                colpack_lib_path1,
                colpack_lib_path2,
                adolc_library_path,
            ]
INCLUDEPATH = [
            colpack_include_path,
            adolc_include_path,
            ]



print ''
print '\033[1;31mplease check that the following settings are correct for your system\033[1;m'
print 'INCLUDEPATH = %s\n'%str(INCLUDEPATH)
print 'LIBPATH = %s\n'%str(LIBPATH)
print '''
If ADOL-C or Colpack cannot be found, you can manually set the paths via
``export ADOLC_DIR=/path/to/adol-c`` and ``export COLPACK_DIR=/path/to/colpack``

* where /path/to/adol-c contains the folders ``ADOL-C/include`` and ``ADOL-C/.libs``.
* where /path/to/colpack contains the folders ``./include`` and ``./lib64``, containing ``libColPack.so`` and the include files

'''
raw_input("Press enter to continue.")



# 0: setup the command line parsing
AddOption('--prefix',
        dest='prefix',
        nargs=1, type='string',
        action='store',
        metavar='DIR',
        help='installation prefix')


env = Environment(
    PREFIX = GetOption('prefix'),
    TMPBUILD = '/tmp/builddir',
    CPPPATH=[distutils.sysconfig.get_python_inc(),numpy.get_include()] + INCLUDEPATH,
    # CXXFLAGS="-ftemplate-depth-100 -DBOOST_PYTHON_DYNAMIC_LIB -O0 -g -Wall",
    CXXFLAGS="-ftemplate-depth-100 -DBOOST_PYTHON_DYNAMIC_LIB -O1 -Wall",
    LIBPATH=  LIBPATH,
    LIBS= LIBS,
    RPATH = LIBPATH, #include information where shared libraries can be found to avoid errors like: "ImportError: libboost_python-gcc42-mt-1_34_1.so.1.34.1: cannot open shared object file: No such file or directory"
    SHLIBPREFIX="", #gets rid of lib prefix
)

Export('env')
Export('adolc_include_path')
SConscript('adolc/SConscript')
SConscript('adolc/sparse/SConscript')
SConscript('adolc/colpack/SConscript')

SConscript('tests/tape_equivalence_PyADOLC_ADOLC/SConscript')


# SConscript('tests/misc_tests/adolc_tiny_unit_test/SConscript')

# env.Install( target='./build/adolc/', source = ['adolc/__init__.py','adolc/wrapped_functions.py', 'adolc/cgraph.py','adolc/_adolc.so'])
#env.Install( target='./build/adolc/sparse/', source = ['adolc/sparse/__init__.py', 'adolc/sparse/_colpack.so', 'adolc/sparse/wrapped_functions.py'])

