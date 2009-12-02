from adolc.tests.test_wrapped_functions import *
from adolc.tests.test_cgraph import *
from adolc.sparse.tests.test_wrapped_functions import *


if __name__ == '__main__':
    try:
        import nose
    except:
        print 'Please install nose for unit testing'
    nose.runmodule()

