import numpy
import numpy.random
from numpy.testing import *
import numpy
import pylab

from adolc import *

class SparseFunctionalityTests(TestCase):


    def test_sparse_jac_no_repeat(self):
        N = 3 # dimension
        M = 2 # codimension
        def vector_f(x):
            return numpy.array([x[0]*x[1],x[1]*x[2]])

        x = numpy.array([1.*n +1. for n in range(N)])
        ax = adouble(x)

        trace_on(1)
        independent(ax)
        ay = vector_f(ax)
        dependent(ay)
        trace_off()

        options = numpy.array([1,1,0,0],dtype=int)
        result = colpack.sparse_jac_no_repeat(1,x,options)
        correct_nnz = 4
        correct_rind   = numpy.array([0,0,1,1])
        corrent_cind   = numpy.array([0,1,1,2])
        correct_values = numpy.array([2.,1.,3.,2.])

        assert_equal(result[0], correct_nnz)
        assert_array_equal(result[1], correct_rind)
        assert_array_equal(result[2], corrent_cind)
        assert_array_almost_equal(result[3], correct_values)

    def test_sparse_jac_with_repeat(self):
        N = 3 # dimension
        M = 2 # codimension
        def vector_f(x):
            return numpy.array([x[0]*x[1],x[1]*x[2]])

        x = numpy.array([1.*n +1. for n in range(N)])
        ax = adouble(x)

        trace_on(1)
        independent(ax)
        ay = vector_f(ax)
        dependent(ay)
        trace_off()

        options = numpy.array([1,1,0,0],dtype=int)

        # first call
        result = colpack.sparse_jac_no_repeat(1,x,options)

        # second call
        x = numpy.array([1.*n +2. for n in range(N)])
        result = colpack.sparse_jac_repeat(1,x, result[0], result[1], result[2], result[3])

        correct_nnz = 4
        correct_rind   = numpy.array([0,0,1,1])
        corrent_cind   = numpy.array([0,1,1,2])
        correct_values = numpy.array([3.,2.,4.,3.])

        assert_equal(result[0], correct_nnz)
        assert_array_equal(result[1], correct_rind)
        assert_array_equal(result[2], corrent_cind)
        assert_array_almost_equal(result[3], correct_values)

    def test_sparse_hess_no_repeat(self):
        N1 = 3 # dimension
        def scalar_f(x):
            return x[0]*x[1] + x[1]*x[2] + x[2]*x[0]

        def scalar_f2(x):
            return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

        x1 = numpy.array([1.*n +1. for n in range(N1)])
        ax1 = adouble(x1)

        trace_on(1)
        independent(ax1)
        ay = scalar_f(ax1)
        dependent(ay)
        trace_off()

        options = numpy.array([0,0],dtype=int)
        result = colpack.sparse_hess_no_repeat(1, x1, options)
        correct_nnz = 3

        correct_rind   = numpy.array([0,0,1])
        corrent_cind   = numpy.array([1,2,2])
        correct_values = numpy.array([1.,1.,1.])

        assert_equal(result[0], correct_nnz)
        assert_array_equal(result[1], correct_rind)
        assert_array_equal(result[2], corrent_cind)
        assert_array_almost_equal(result[3], correct_values)


        N2 = 4
        x2 = numpy.array([1.*n +1. for n in range(N2)])

        trace_on(2)
        ax2 = adouble(x2)
        independent(ax2)
        ay = scalar_f2(ax2)
        dependent(ay)
        trace_off()

        options = numpy.array([0,0],dtype=int)
        for i in range(10):
            result = colpack.sparse_hess_no_repeat(2, x2, options)

    def test_sparse_hess_repeat(self):
        N = 3 # dimension

        def scalar_f(x):
            return x[0]**3 + x[0]*x[1] + x[1]*x[2] + x[2]*x[0]

        x = numpy.array([1.*n +1. for n in range(N)])
        ax = adouble(x)

        trace_on(1)
        independent(ax)
        ay = scalar_f(ax)
        dependent(ay)
        trace_off()

        options = numpy.array([1,1],dtype=int)

        # first call
        result = colpack.sparse_hess_no_repeat(1,x,options)

        # second call
        x = numpy.array([1.*n +2. for n in range(N)])
        result = colpack.sparse_hess_repeat(1,x, result[1], result[2], result[3])

        correct_nnz = 4

        correct_rind   = numpy.array([0,0,0,1])
        corrent_cind   = numpy.array([0,1,2,2])
        correct_values = numpy.array([6*x[0],1.,1.,1.])

        assert_equal(result[0], correct_nnz)
        assert_array_equal(result[1], correct_rind)
        assert_array_equal(result[2], corrent_cind)
        assert_array_almost_equal(result[3], correct_values)

    def test_sparse_problem(self):
        return 0
        import scipy.sparse

        nvar = 4
        ncon = 2

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


        x0 = numpy.array([1.0, 5.0, 5.0, 1.0])

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
            result = colpack.sparse_jac_no_repeat(2,x,options)
            if flag:
                return (numpy.asarray(result[1],dtype=int), numpy.asarray(result[2],dtype=int))
            else:
                return result[3]

        def eval_h_adolc(x, lagrange, obj_factor, flag, user_data = None):
            options = numpy.array([0,0],dtype=int)
            assert numpy.ndim(x) == 1
            assert numpy.size(x) == 4
            result_f = colpack.sparse_hess_no_repeat(1, x, options)
            result_g0 = colpack.sparse_hess_no_repeat(3, x,options)
            result_g1 = colpack.sparse_hess_no_repeat(4, x,options)
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


if __name__ == '__main__':
    try:
        import nose
    except:
        print 'Please install nose for unit testing'
    nose.runmodule()

