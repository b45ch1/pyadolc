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
    
    
    
if __name__ == "__main__":
    test_big_tape()
