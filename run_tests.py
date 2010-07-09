from adolc.tests.test_wrapped_functions import *
from adolc.tests.test_wrapped_functions_complicated import *
from adolc.tests.test_wrapped_functions_future import *

from adolc.tests.test_cgraph import *


if __name__ == '__main__':
    try:
        import nose
    except:
        print 'Please install nose for unit testing'
    nose.runmodule()

