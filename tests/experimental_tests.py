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
    



def test_ipopt_optimization():
    """
    This test checks
    1. the sparse functionality of pyadolc
    2. the execution speed compared to the direct sparse computation
    3. run the optimization with the derivatives provided by pyadolc
    
    IPOPT is an interior point algorithm to solve
    
    min     f(x)
        x in R^n
    s.t.       g_L <= g(x) <= g_U
                x_L <=  x   <= x_U
    
    this test fails probably because of a bug in pyipopt 

    """
    
    try:
        import pyipopt
    except:
        #print '"pyipopt is not installed, skipping test'
        #return
        raise NotImplementedError("pyipopt is not installed, skipping test")
    import time

    nvar = 4
    x_L = numpy.ones((nvar), dtype=numpy.float_) * 1.0
    x_U = numpy.ones((nvar), dtype=numpy.float_) * 5.0

    ncon = 2
    g_L = numpy.array([25.0, 40.0])
    g_U = numpy.array([2.0*pow(10.0, 19), 40.0]) 

    def eval_f(x, user_data = None):
        assert len(x) == 4
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def eval_grad_f(x, user_data = None):
        assert len(x) == 4
        grad_f = numpy.array([
            x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]) ,
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2])
            ])
        return grad_f;
        
    def eval_g(x, user_data= None):
        assert len(x) == 4
        return numpy.array([
            x[0] * x[1] * x[2] * x[3], 
            x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]
        ])

    nnzj = 8
    def eval_jac_g(x, flag, user_data = None):
        if flag:
            return (numpy.array([0, 0, 0, 0, 1, 1, 1, 1]), 
                numpy.array([0, 1, 2, 3, 0, 1, 2, 3]))
        else:
            assert len(x) == 4
            return numpy.array([ x[1]*x[2]*x[3], 
                        x[0]*x[2]*x[3], 
                        x[0]*x[1]*x[3], 
                        x[0]*x[1]*x[2],
                        2.0*x[0], 
                        2.0*x[1], 
                        2.0*x[2], 
                        2.0*x[3] ])
            
    nnzh = 10
    def eval_h(x, lagrange, obj_factor, flag, user_data = None):
        if flag:
            hrow = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
            hcol = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
            return (numpy.array(hcol,dtype=int), numpy.array(hrow,dtype=int))
        else:
            values = numpy.zeros((10), numpy.float_)
            values[0] = obj_factor * (2*x[3])
            values[1] = obj_factor * (x[3])
            values[2] = 0
            values[3] = obj_factor * (x[3])
            values[4] = 0
            values[5] = 0
            values[6] = obj_factor * (2*x[0] + x[1] + x[2])
            values[7] = obj_factor * (x[0])
            values[8] = obj_factor * (x[0])
            values[9] = 0
            values[1] += lagrange[0] * (x[2] * x[3])

            values[3] += lagrange[0] * (x[1] * x[3])
            values[4] += lagrange[0] * (x[0] * x[3])

            values[6] += lagrange[0] * (x[1] * x[2])
            values[7] += lagrange[0] * (x[0] * x[2])
            values[8] += lagrange[0] * (x[0] * x[1])
            values[0] += lagrange[1] * 2
            values[2] += lagrange[1] * 2
            values[5] += lagrange[1] * 2
            values[9] += lagrange[1] * 2
            return values



    def apply_new(x):
        return True

    x0 = numpy.array([1.0, 5.0, 5.0, 1.0])
    pi0 = numpy.array([1.0, 1.0])

    # check that adolc gives the same answers as derivatives calculated by hand
    trace_on(1)
    ax = adouble(x0)
    independent(ax)
    ay = eval_f(ax)
    dependent(ay)
    trace_off()

    trace_on(2)
    ax = adouble(x0)
    independent(ax)
    ay = eval_g(ax)
    dependent(ay)
    trace_off()
    
    trace_on(3)
    ax = adouble(x0)
    independent(ax)
    ay = eval_g(ax)
    dependent(ay[0])
    trace_off()
    
    trace_on(4)
    ax = adouble(x0)
    independent(ax)
    ay = eval_g(ax)
    dependent(ay[1])
    trace_off()
    

    def eval_f_adolc(x, user_data = None):
        return function(1,x)[0]

    def eval_grad_f_adolc(x, user_data = None):
        return gradient(1,x)

    def eval_g_adolc(x, user_data= None):
        return function(2,x)

    def eval_jac_g_adolc(x, flag, user_data = None):
        options = numpy.array([1,1,0,0],dtype=int)
        result = sparse.sparse_jac_no_repeat(2,x,options)
        if flag:
            return (numpy.asarray(result[1],dtype=int), numpy.asarray(result[2],dtype=int))
        else:
            return result[3]
            
    def eval_h_adolc(x, lagrange, obj_factor, flag, user_data = None):
        options = numpy.array([0,0],dtype=int)
        assert numpy.ndim(x) == 1
        assert numpy.size(x) == 4
        result_f = sparse.sparse_hess_no_repeat(1, x, options)
        result_g0 = sparse.sparse_hess_no_repeat(3, x,options)
        result_g1 = sparse.sparse_hess_no_repeat(4, x,options)
        Hf  = scipy.sparse.coo_matrix( (result_f[3], (result_f[1], result_f[2])), shape=(4, 4))
        Hg0 = scipy.sparse.coo_matrix( (result_g0[3], (result_g0[1], result_g0[2])), shape=(4, 4))
        Hg1 = scipy.sparse.coo_matrix( (result_g1[3], (result_g1[1], result_g1[2])), shape=(4, 4))
        
        H = Hf + Hg0 + Hg1
        H = H.tocoo()
        
        if flag:
            hrow = H.row
            hcol = H.col
            return (numpy.array(hcol,dtype=int), numpy.array(hrow,dtype=int))

        else:
            values = numpy.zeros((10), float)
            values[:] = H.data
            return values

    # function of f
    assert_almost_equal(eval_f(x0), eval_f_adolc(x0))
    
    # gradient of f
    assert_array_almost_equal(eval_grad_f(x0), eval_grad_f_adolc(x0))

    # function of g
    assert_array_almost_equal(eval_g(x0), function(2,x0))

    # sparse jacobian of g
    assert_array_equal(eval_jac_g_adolc(x0,True)[0], eval_jac_g(x0,True)[0])
    assert_array_equal(eval_jac_g_adolc(x0,True)[1], eval_jac_g(x0,True)[1])
    assert_array_equal(eval_jac_g_adolc(x0,False),  eval_jac_g(x0,False))
    
    # sparse hessian of the lagrangian
    lagrange = numpy.ones(2,dtype=float)
    obj_factor = 1.
    x0 = numpy.random.rand(4)
    result       = (eval_h(x0, lagrange, obj_factor, False), eval_h(x0, lagrange, obj_factor, True))
    result_adolc = (eval_h_adolc(x0, lagrange, obj_factor, False), eval_h_adolc(x0, lagrange, obj_factor, True))
    H       = scipy.sparse.coo_matrix( result, shape=(4, 4))
    H_adolc = scipy.sparse.coo_matrix( result_adolc, shape=(4, 4))
    H = H.todense()
    H_adolc = H_adolc.todense()
    assert_array_almost_equal( H, H_adolc.T)


    # test optimization with PYIPOPT
    nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h)
    start_time = time.time()
    result =  nlp.solve(x0)
    end_time = time.time()
    nlp.close()
    pure_python_optimization_time = end_time - start_time


    nlp_adolc = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f_adolc, eval_grad_f_adolc, eval_g_adolc, eval_jac_g_adolc, eval_h_adolc)

    start_time = time.time()
    result_adolc = nlp_adolc.solve(x0)
    end_time = time.time()
    nlp_adolc.close()
    
    adolc_optimization_time = end_time - start_time
    print 'optimization time with derivatives computed by adolc = ', adolc_optimization_time
    print 'optimization time with derivatives computed by hand = ',pure_python_optimization_time
    assert adolc_optimization_time / pure_python_optimization_time < 10
    
    # this works with the pyipopt version from code.google.com
    assert_array_almost_equal(result[0], result_adolc[0])
    assert_array_almost_equal(result[1], result_adolc[1])
    assert_array_almost_equal(result[2], result_adolc[2])
    assert_array_almost_equal(result[3], result_adolc[3])
    
    ##this works with the pyipopt version from github by alanfalloon
    #assert_array_almost_equal(result['x'],result_adolc['x'])
    #assert_array_almost_equal(result['mult_xL'],result_adolc['mult_xL'])
    #assert_array_almost_equal(result['mult_xU'],result_adolc['mult_xU'])
    #assert_array_almost_equal(result['mult_g'],result_adolc['mult_g'])
    #assert_array_almost_equal(result['f'],result_adolc['f'])


if __name__ == '__main__':
    try:
        import nose
    except:
        print 'Please install nose for unit testing'
    nose.runmodule()