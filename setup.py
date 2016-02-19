#!/usr/bin/env python

""" PYADOLC, Python Bindings to ADOL-C
"""
DOCLINES = __doc__.split("\n")

# build with: $ python setup.py build_ext --inplace
# clean with: # python setup.py clean --all
# see:
# http://www.scipy.org/Documentation/numpy_distutils
# http://docs.cython.org/docs/tutorial.html

import os
from distutils.core import setup, Extension
from distutils.core import Command
from numpy.distutils.misc_util import get_numpy_include_dirs
import inspect


BASEDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

BOOST_DIR   = os.environ.get('BOOST_DIR', os.path.join(BASEDIR, '/usr/local'))
ADOLC_DIR   = os.environ.get('ADOLC_DIR', os.path.join(BASEDIR, 'PACKAGES/ADOL-C/inst'))
COLPACK_DIR = os.environ.get('COLPACK_DIR', os.path.join(BASEDIR, 'PACKAGES/ADOL-C/ThirdParty/ColPack'))

boost_include_path   = os.path.join(BOOST_DIR, 'include')
boost_library_path1  = os.path.join(BOOST_DIR, 'lib')
boost_library_path2  = os.path.join(BOOST_DIR, 'lib64')

adolc_include_path   = os.path.join(ADOLC_DIR, 'include')
adolc_library_path1  = os.path.join(ADOLC_DIR, 'lib')
adolc_library_path2  = os.path.join(ADOLC_DIR, 'lib64')

colpack_include_path = os.path.join(COLPACK_DIR, 'include')
colpack_lib_path1    = os.path.join(COLPACK_DIR, 'lib')
colpack_lib_path2    = os.path.join(COLPACK_DIR, 'lib64')

# ADAPT THIS TO FIT YOUR SYSTEM
extra_compile_args = ['-std=c++11 -ftemplate-depth-100 -DBOOST_PYTHON_DYNAMIC_LIB']
include_dirs = [get_numpy_include_dirs()[0], boost_include_path, adolc_include_path, colpack_include_path]
library_dirs = [boost_library_path1, boost_library_path2, adolc_library_path1, adolc_library_path2, colpack_lib_path1, colpack_lib_path2]
libraries = ['boost_python','adolc', 'ColPack']

print ''
print '\033[1;31m Note: If this script does not work you can try to use scons.\033[1;m'
print '\033[1;31mplease check that the following settings are correct for your system\033[1;m'
print 'include_dirs = %s\n'%str(include_dirs)
print 'library_dirs = %s\n'%str(library_dirs)
print '''
If ADOL-C or Colpack cannot be found, you can manually set the paths via
``export ADOLC_DIR=/path/to/adol-c`` and ``export COLPACK_DIR=/path/to/colpack``

* where /path/to/adol-c contains the folders ``./include`` and ``./lib64``.
* where /path/to/colpack contains the folders ``./include`` and ``./lib64``, containing ``libColPack.so`` and the include files

'''
raw_input("Press enter to continue.")


# PACKAGE INFORMATION
CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: C++
Programming Language :: Python
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Linux
"""

NAME                = 'pyadolc'
MAINTAINER          = "Sebastian F. Walter"
MAINTAINER_EMAIL    = "sebastian.walter@gmail.com"
DESCRIPTION         = DOCLINES[0]
LONG_DESCRIPTION    = "\n".join(DOCLINES[2:])
URL                 = "http://www.github.com/b45ch1/pyadolc"
DOWNLOAD_URL        = "http://www.github.com/b45ch1/pyadolc"
LICENSE             = 'BSD'
CLASSIFIERS         = filter(None, CLASSIFIERS.split('\n'))
AUTHOR              = "Sebastian F. Walter"
AUTHOR_EMAIL        = "sebastian.walter@gmail.com"
PLATFORMS           = ["Linux"]
MAJOR               = 0
MINOR               = 1
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

# IT IS USUALLY NOT NECESSARY TO CHANGE ANTHING BELOW THIS POINT
# override default setup.py help output
import sys
if len(sys.argv) == 1:
    print """

    You didn't enter what to do!

    Options:
    1: build the extension with
    python setup.py build

    2: install the extension with
    python setup.py install

    3: alternatively build inplace
    python setup.py build_ext --inplace

    4: remove generated files with
    python setup.py clean --all


    Remark: This is an override of the default behaviour of the distutils setup.
    """
    exit()

class clean(Command):
    """
    This class is used in numpy.distutils.core.setup.
    When $python setup.py clean is called, an instance of this class is created and then it's run method is called.
    """

    description = "Clean everything"
    user_options = [("all","a","the same")]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        import os
        os.system("rm -rf build")
        os.system("rm _adolc.so")
        os.system("rm -f py_adolc.os num_util.os")
        os.system("rm *.pyc")


def fullsplit(path, result=None):
    """
    Split a pathname into components (the opposite of os.path.join) in a
    platform-neutral way.
    """
    if result is None:
        result = []
    head, tail = os.path.split(path)
    if head == '':
        return [tail] + result
    if head == path:
        return result
    return fullsplit(head, [tail] + result)

# find all files that should be included
packages, data_files = [], []
for dirpath, dirnames, filenames in os.walk('adolc'):
    # Ignore dirnames that start with '.'
    for i, dirname in enumerate(dirnames):
        if dirname.startswith('.'): del dirnames[i]
    if '__init__.py' in filenames:
        packages.append('.'.join(fullsplit(dirpath)))
    elif filenames:
        data_files.append([dirpath, [os.path.join(dirpath, f) for f in filenames]])

options_dict = {}
options_dict.update({
'name':NAME,
'version':VERSION,
'description' :DESCRIPTION,
'long_description' : LONG_DESCRIPTION,
'license':LICENSE,
'author':AUTHOR,
'platforms':PLATFORMS,
'author_email': AUTHOR_EMAIL,
'url':URL,
'packages' :packages,
'ext_package' : 'adolc',
'ext_modules': [Extension('_adolc', ['adolc/src/py_adolc.cpp',  'adolc/src/py_interpolation.cpp', 'adolc/src/num_util.cpp'],
                                include_dirs = ['adolc/src'] + include_dirs,
                                library_dirs = library_dirs,
                                runtime_library_dirs = library_dirs,
                                libraries = libraries,
                                extra_compile_args = extra_compile_args),
                Extension('sparse/_sparse', ['adolc/sparse/src/py_sparse_adolc.cpp', 'adolc/sparse/src/num_util.cpp'],
                                include_dirs = ['adolc/sparse/src'] + include_dirs,
                                library_dirs = library_dirs,
                                runtime_library_dirs = library_dirs,
                                libraries = libraries,
                                extra_compile_args = extra_compile_args),
                Extension('colpack/_colpack', ['adolc/colpack/src/py_colpack_adolc.cpp', 'adolc/colpack/src/num_util.cpp'],
                                include_dirs = ['adolc/colpack/src'] + include_dirs,
                                library_dirs = library_dirs,
                                runtime_library_dirs = library_dirs,
                                libraries = libraries,
                                extra_compile_args = extra_compile_args),
],

'cmdclass' : {'clean':clean}
})

setup(**options_dict)
