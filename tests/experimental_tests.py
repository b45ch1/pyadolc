"""
This file is supposed to be used for "tests as you go". I.e. you specify a feature that should be tested,
but is not implemented correctly yet.
Once, the feature is correctly implemented and passes this test, it should be moved to either
unit_tests.py
or complicated_tests.py
"""
import numpy
from adolc import *

def test_big_tape():
    """
    Hmm, this should raise an error because num_max_lives is bigger than the value buffer
    """
    N = 65534*4

    ay = adouble(0.)
    ax = adouble(0.)
    x = 1.

    # usual way: leads to increasing locints
    trace_on(0)
    independent(ax)
    ay = ax
    for i in range(N):
        ay = ay * ay

    dependent(ay)
    trace_off()

    print tapestats(0)
    print function(0,[1.])
    #tape_to_latex(0,numpy.array([1.]),numpy.array([0.]))


def test_arc_hyperbolic_functions():
    x = 3.
    ax = adouble(x)
    
    aarcsh = numpy.arcsinh(ax)
    aarcch = numpy.arccosh(ax)
    aarcth = numpy.arctanh(ax)
    
    assert_almost_equal(aarcsh.val, numpy.arcsinh(x))
    assert_almost_equal(aarcch.val, numpy.arccosh(x))
    assert_almost_equal(aarcth.val, numpy.arctanh(x))
    
    
if __name__ == '__main__':
    try:
        import nose
    except:
        print 'Please install nose for unit testing'
    nose.runmodule()