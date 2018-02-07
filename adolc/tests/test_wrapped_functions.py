import numpy
import numpy.random
from numpy.testing import *
import numpy
import pylab

from adolc import *

class OperationsTests ( TestCase ):

    def test_constructors(self):
        a = adouble(13.);
        b = adouble(5)
        c = adouble(a)

        assert a.val == 13.
        assert b.val == 5
        assert c.val == 13.

    def test_unary_operators(self):
        a = adouble(1.)
        b = -a
        assert b.val == -1.
        assert a.val == 1.

        print type(b)
        print type(a)

    def test_conditional_operators(self):
        ax = adouble(2.)
        ay = adouble(1.)

        assert ax <= 2
        assert ax <= 2.
        assert not ax < 2
        assert not ax < 2.


        assert ax >= 2
        assert ax >= 2.
        assert not ax > 2
        assert not ax > 2.

        assert ax >  ay
        assert ax >= ay
        assert not ax <  ay
        assert not ax <= ay


    def test_radd(self):
        a = adouble(1.)
        b = a + 2.
        c = a + 2.
        d = 2.+ a

        assert a.val == 1.

    def test_add(self):
        a = adouble(1.)
        b = a + 2.
        c = a + 2
        d = 2.+ a
        e = 2 + a

        assert b.val == 3.
        assert c.val == 3.
        assert d.val == 3.
        assert e.val == 3.

    def test_sub(self):
        a = adouble(1.)
        b = a - 2.
        c = 2.- a

        assert b.val == -1.
        assert c.val == 1.

    def test_mul(self):
        a = adouble(1.5)
        b = a * 2.
        c = 2.* a

        assert b.val == 3.
        assert c.val == 3.

    def test_div(self):
        a = adouble(3.)
        b = a/2.
        c = 2./a

        assert b.val == 3./2.
        assert c.val == 2./3.

    def test_truediv(self):
        x=1
        y=2
        ax=adouble(x)
        ay=adouble(y)

        z= x.__truediv__(y)
        az1=ax.__truediv__(y)
        az2=ay.__rtruediv__(x)
        az3=ax.__truediv__(ay)

        assert_almost_equal(az1.val, z)
        assert_almost_equal(az2.val, z)
        assert_almost_equal(az3.val, z)

    def test_pow(self):
        r  = 5
        x = 3.
        y = 2.
        ax = adouble(x)
        ay = adouble(y)

        az1 = ax**ay
        az2 = ax**r
        az3 = r**ax

        assert_almost_equal(az1.val, x**y)
        assert_almost_equal(az2.val, x**r)
        assert_almost_equal(az3.val, r**x)

    def test_hyperbolic_functions(self):
        x = 3.
        ax = adouble(x)

        ash = numpy.sinh(ax)
        ach = numpy.cosh(ax)
        ath = numpy.tanh(ax)

        assert_almost_equal(ash.val, numpy.sinh(x))
        assert_almost_equal(ach.val, numpy.cosh(x))
        assert_almost_equal(ath.val, numpy.tanh(x))

    def test_arc_functions(self):

        x = 0.65
        y = 2.1

        ax = adouble(x)
        ay = adouble(y)

        aas = numpy.arcsin(ax)
        aac = numpy.arccos(ax)
        aat = numpy.arctan(ax)
        aat2 = numpy.arctan2(ax, ay)

        assert_almost_equal(aas.val, numpy.arcsin(x))
        assert_almost_equal(aac.val, numpy.arccos(x))
        assert_almost_equal(aat.val, numpy.arctan(x))
        assert_almost_equal(aat2.val, numpy.arctan2(x,y))

    def test_fabs(self):
        x = 3.
        xs = numpy.array([1.,2.,3.])
        ax = adouble(x)
        axs = adouble(xs)

        aabs = numpy.fabs(ax)
        aabss = numpy.fabs(axs)

        assert_almost_equal(aabs.val, numpy.fabs(x))

    def test_abs(self):
        x = 3.
        xs = numpy.array([1.,2.,3.])
        ax = adouble(x)
        axs = adouble(xs)

        afabs = abs(ax)
        afabss = abs(axs)

        assert_almost_equal(afabs.val, abs(x))

    def test_numpyabs(self):
        x = 3.
        xs = numpy.array([1.,2.,3.])
        ax = adouble(x)
        axs = adouble(xs)

        afabs = numpy.abs(ax)
        afabss = numpy.abs(axs)

        assert_almost_equal(afabs.val, numpy.abs(x))

        #test_expression('fabs (a)     : ',		lambda x: numpy.fabs (x),  a,		a.val)


    def test_double_condassign_if(self):
        x = 3.
        y = 4.
        cond = 1.

        x = condassign(x,cond,y)
        print x
        assert_almost_equal(x,4.)

        x = 3.
        y = 4.
        cond = -1.
        x = condassign(x,cond,y)
        print x
        assert_almost_equal(x,3.)

    def test_double_condassign_if_else(self):
        x = 3.
        y = 4.
        z = 5.
        cond = 1.

        x = condassign(x,cond,y,z)
        assert x == 4.

        x = 3.
        y = 4.
        z = 5.
        cond = -1.

        x = condassign(x,cond,y,z)
        assert x == 5

    def test_double_condeqassign_if(self):
        x = 3.
        y = 4.
        cond = 0.

        x = condeqassign(x,cond,y)
        print x
        assert_almost_equal(x,4.)

        x = 3.
        y = 4.
        cond = -1.
        x = condeqassign(x,cond,y)
        print x
        assert_almost_equal(x,3.)

    def test_double_condeqassign_if_else(self):
        x = 3.
        y = 4.
        z = 5.
        cond = 0.

        x = condeqassign(x,cond,y,z)
        assert x == 4.

        x = 3.
        y = 4.
        z = 5.
        cond = -1.

        x = condeqassign(x,cond,y,z)
        assert x == 5

    def test_adouble_condassign_if(self):
        trace_on(1)
        x = independent(adouble(3.))
        y = adouble(4.)
        cond = adouble(x-1.)
        r = dependent(condassign(x,cond,y))
        trace_off()
        assert_almost_equal(r.val, 4.)
        y_arr = function(1,numpy.array([3.0]))
        assert_almost_equal(y_arr[0],4.0)
        y_arr = function(1,numpy.array([1.0]))
        assert_almost_equal(y_arr[0],1.0)
        y_arr = function(1,numpy.array([.5]))
        assert_almost_equal(y_arr[0],.5)

        x = adouble(3.)
        y = adouble(4.)
        cond = adouble(-3.)
        x = condassign(x,cond,y)
        print x
        assert_almost_equal(x.val, 3.)

    def test_adouble_condeqassign_if(self):
        trace_on(1)
        x = independent(adouble(3.))
        y = adouble(4.)
        cond = adouble(x-1.)
        r = dependent(condeqassign(x,cond,y))
        trace_off()
        assert_almost_equal(r.val, 4.)
        y_arr = function(1,numpy.array([3.0]))
        assert_almost_equal(y_arr[0],4.0)
        y_arr = function(1,numpy.array([1.0]))
        assert_almost_equal(y_arr[0],4.0)
        y_arr = function(1,numpy.array([.5]))
        assert_almost_equal(y_arr[0],.5)

        x = adouble(3.)
        y = adouble(4.)
        cond = adouble(-3.)
        x = condeqassign(x,cond,y)
        print x
        assert_almost_equal(x.val, 3.)

    def test_xuchen_condassign(self):
        """
        see https://github.com/b45ch1/pyadolc/issues/12
        """

        def f(x):
             a = x + 1
             b = 2*x
             x = condassign(x, b-a, b)
             return x


        def g(x):
             a = x + 1
             b = 2*x
             if b-a > 0:
                x = b
             return x


        trace_on(0)
        ax = adouble(2.0)
        independent(ax)
        ay = f(ax)
        dependent(ay)
        trace_off()

        assert_array_almost_equal(g(-1.), zos_forward(0, -1., keep=0)[0])
        assert_array_almost_equal(g(1.), zos_forward(0, 1., keep=0)[0])
        assert_array_almost_equal(g(2.), zos_forward(0, 2., keep=0)[0])

    def test_adouble_condassign_if_else(self):
        x = adouble(3.)
        y = adouble(4.)
        z = adouble(5.)
        cond = adouble(1.)

        x = condassign(x,cond,y,z)
        print x
        assert_almost_equal(x.val, 4.)

        x = adouble(3.)
        y = adouble(4.)
        z = adouble(5.)
        cond = adouble(-3.)

        x = condassign(x,cond,y,z)
        print x
        assert_almost_equal(x.val, 5.)

    def test_adouble_condeqassign_if_else(self):
        x = adouble(3.)
        y = adouble(4.)
        z = adouble(5.)
        cond = adouble(0.)

        x = condeqassign(x,cond,y,z)
        print x
        assert_almost_equal(x.val, 4.)

        x = adouble(3.)
        y = adouble(4.)
        z = adouble(5.)
        cond = adouble(-3.)

        x = condeqassign(x,cond,y,z)
        print x
        assert_almost_equal(x.val, 5.)

class CorrectnessTests(TestCase):
    def test_sin(self):
        def eval_f(x):
            return numpy.sin(x[0] + x[1]*x[0])

        def eval_g(x):
            return numpy.array([numpy.cos(x[0] + x[1]*(1 + x[1])),
                                numpy.cos(x[0] + x[1]*x[0])
                                ])

        #tape f
        ax = numpy.array([adouble(0.) for i in range(3)])
        trace_on(0)
        for i in range(2):
            independent(ax[i])
        ay = eval_f(ax)
        dependent(ay)
        trace_off()

        x = numpy.array([3.,7.])

        g = gradient(0,x)

        print g
        print eval_g(x)


class LowLevelFunctionsTests ( TestCase ):

    def test_independent(self):
        # 0D
        ax = adouble(1)
        bx = independent(ax)
        assert ax == bx

        # 1D
        N = 10
        ax = numpy.array([adouble(n) for n in range(N)])
        bx = independent(ax)
        assert numpy.prod( ax == bx )

        # 2D
        N = 2; M=3
        ax = numpy.array([[adouble(n+m) for n in range(N)] for m in range(M)])
        bx = independent(ax)
        assert numpy.prod( ax == bx )

    def test_dependent(self):
        # 0D
        ax = adouble(1)
        bx = dependent(ax)
        assert ax == bx

        # 1D
        N = 10
        ax = numpy.array([adouble(n) for n in range(N)])
        bx = dependent(ax)
        assert numpy.prod( ax == bx )

        # 2D
        N = 2; M=3
        ax = numpy.array([[adouble(n+m) for n in range(N)] for m in range(M)])
        bx = dependent(ax)
        assert numpy.prod( ax == bx )


    def test_hos_forward_with_keep_then_hos_ti_reverse(self):
        """compute the first columnt of the hessian of f = x_1 x_2 x_3"""
        def f(x):
            return x[0]*x[1]*x[2]

        #tape f
        ax = numpy.array([adouble(0.) for i in range(3)])
        trace_on(0)
        for i in range(3):
            independent(ax[i])
        ay = f(ax)
        dependent(ay)
        trace_off()

        x = numpy.array([3.,5.,7.])
        V = numpy.zeros((3,1))
        V[0,0]=1

        (y,W) = hos_forward(0,x,V,2)
        assert y[0] == 105.
        assert W[0] == 35.

        U = numpy.zeros((1,2), dtype=float)
        U[0,0] = 1.

        Z = hos_ti_reverse(0,U)
        assert numpy.prod( Z[:,0] == numpy.array([35., 21., 15.]))
        assert numpy.prod( Z[:,1] == numpy.array([0., 7., 5.]))


    def test_hov_ti_reverse(self):
        """compute the first columnt of the hessian of f = x_1 x_2 x_3"""
        def f(x):
            return x[0]*x[1]*x[2]

        #tape f
        ax = numpy.array([adouble(0.) for i in range(3)])
        trace_on(0)
        for i in range(3):
            independent(ax[i])
        ay = f(ax)
        dependent(ay)
        trace_off()

        x = numpy.array([3.,5.,7.])
        V = numpy.zeros((3,1))
        V[0,0]=1

        (y,W) = hos_forward(0,x,V,2)
        assert y[0] == 105.
        assert W[0] == 35.

        U = numpy.zeros((1,1,2), dtype=float)
        U[0,0,0] = 1.

        Z = hov_ti_reverse(0,U)[0]
        print Z[0,:,0]
        assert numpy.prod( Z[0,:,0] == numpy.array([35., 21., 15.]))
        assert numpy.prod( Z[0,:,1] == numpy.array([0., 7., 5.]))


    def test_hov_wk_forward_with_keep_then_hos_ov_reverse(self):
        """compute the full hessian of f = x_1 x_2 x_3"""
        def f(x):
            return x[0]*x[1]*x[2]

        #tape f
        ax = numpy.array([adouble(0.) for i in range(3)])
        trace_on(0)
        for i in range(3):
            independent(ax[i])
        ay = f(ax)
        dependent(ay)
        trace_off()

        x = numpy.array([3.,5.,7.])
        P = 3
        V = numpy.zeros((3,P,1))
        V[0,0,0] = 1.
        V[1,1,0] = 1.
        V[2,2,0] = 1.

        (y,W) = hov_wk_forward(0,x,V,2)
        assert_almost_equal(y[0],105.)

        U = numpy.zeros((1,2), dtype=float)
        U[0,0] = 1.

        Z = hos_ov_reverse(0, P ,U)

        H = numpy.array([[0, x[2], x[1]],[x[2], 0, x[0]], [x[1], x[0],0]],dtype=float)
        assert_array_almost_equal(Z[:,:,1],H)



    def test_simple_function(self):
        def f(x):
            y1 = 1./(numpy.fabs(x))
            y2 = x*5.
            y3 = y1 + y2
            return y3
        def g(x):
            return -1./numpy.fabs(x)**2 + 5.

        #tape f
        trace_on(0)
        x = 2.
        ax = adouble(x)
        independent(ax)
        ay = f(ax)
        depends_on(ay)
        trace_off()
        assert_array_almost_equal(g(x), gradient(0,numpy.array([x])))

    def test_tape_to_latex(self):
        N = 40
        def scalar_f(x):
            return 0.5*numpy.dot(x,x)

        x = numpy.array([1.*n for n in range(N)])
        ax = adouble(x)

        trace_on(123)
        independent(ax)
        ay = scalar_f(ax)
        dependent(ay)
        trace_off()
        y = numpy.zeros(1)
        tape_to_latex(123,x,y)
        import os
        os.system("mv tape_123.tex /tmp")
        cwd = os.getcwd()
        os.chdir("/tmp")
        os.system("pdflatex tape_123.tex ")
        os.chdir(cwd)


class HighLevelFunctionsTests ( TestCase ):
    """
    TESTING HIGH LEVEL CONVENICENCE FUNCTIONS (GRADIENT,HESSIAN, ETC..)
    """

    def test_function(self):
        N = 10
        def scalar_f(x):
            return numpy.dot(x,x)

        x = numpy.ones(N)
        ax = adouble(x)

        trace_on(0)
        independent(ax)
        ay = scalar_f(ax)
        dependent(ay)
        trace_off()
        assert_almost_equal(scalar_f(x),function(0,x))

    def test_gradient(self):
        N = 10
        def scalar_f(x):
            return 0.5*numpy.dot(x,x)

        x = numpy.array([1.*n for n in range(N)])
        ax = adouble(x)

        trace_on(0)
        independent(ax)
        ay = scalar_f(ax)
        dependent(ay)
        trace_off()
        assert_array_almost_equal(x,gradient(0,x))

    def test_hessian(self):
        N = 10
        def scalar_f(x):
            return 0.5*numpy.dot(x,x)

        x = numpy.array([1.*n for n in range(N)])
        ax = adouble(x)

        trace_on(0)
        independent(ax)
        ay = scalar_f(ax)
        dependent(ay)
        trace_off()
        true_H = numpy.eye(N)
        assert_array_almost_equal(true_H, hessian(0,x))

    def test_jacobian(self):
        N = 31 # dimension
        M = 29 # codimension
        A = numpy.array([[ 1./N +(n==m) for n in range(N)] for m in range(M)])
        def vector_f(x):
            return numpy.dot(A,x)

        x = numpy.array([1.*n for n in range(N)])
        ax = adouble(x)

        trace_on(123)
        independent(ax)
        ay = vector_f(ax)
        dependent(ay)
        trace_off()
        assert_array_almost_equal(A, jacobian(123,x))

    def test_hess_vec(self):
        N = 1132
        def scalar_f(x):
            return 0.5*numpy.dot(x,x)

        x = numpy.array([1.*n for n in range(N)])
        ax = adouble(x)

        trace_on(0)
        independent(ax)
        ay = scalar_f(ax)
        dependent(ay)
        trace_off()

        v = numpy.random.rand(N)
        H = numpy.eye(N)
        Hv = numpy.dot(H,v)
        assert_array_almost_equal( Hv, hess_vec(0,x,v))

    def test_vec_jac(self):
        N = 3 # dimension
        M = 2 # codimension
        A = numpy.array([[ 1./N +(n==m) for n in range(N)] for m in range(M)])
        def vector_f(x):
            return numpy.dot(A,x)

        x = numpy.array([1.*n for n in range(N)])
        ax = adouble(x)

        trace_on(1)
        independent(ax)
        ay = vector_f(ax)
        dependent(ay)
        trace_off()
        u = numpy.random.rand(M)
        uJ = numpy.dot(u,A)
        assert_array_almost_equal( uJ, vec_jac(1,x,u, 0))


    def test_jac_vec(self):
        N = 3 # dimension
        M = 2 # codimension
        A = numpy.array([[ 1./N +(n==m) for n in range(N)] for m in range(M)])
        def vector_f(x):
            return numpy.dot(A,x)

        x = numpy.array([1.*n for n in range(N)])
        ax = adouble(x)

        trace_on(1)
        independent(ax)
        ay = vector_f(ax)
        dependent(ay)
        trace_off()
        v = numpy.random.rand(N)
        Jv = numpy.dot(A,v)
        assert_array_almost_equal( Jv, jac_vec(1,x,v) )

    def test_lagra_hess_vec(self):
        """ This test needs improvement: the result is always 0!!"""
        N = 3 # dimension
        M = 2 # codimension
        A = numpy.array([[ 1./N +(n==m) for n in range(N)] for m in range(M)])
        def vector_f(x):
            return numpy.dot(A,x)

        x = numpy.array([1.*n for n in range(N)])
        ax = adouble(x)

        trace_on(1)
        independent(ax)
        ay = vector_f(ax)
        dependent(ay)
        trace_off()
        u = numpy.random.rand(M)
        v = numpy.random.rand(N)
        assert_array_almost_equal(numpy.zeros(N,dtype=float), lagra_hess_vec(1,x,u,v) )

    def test_repeated_taping(self):
        R = 20 # number of repetitions of the taping

        N = 3 # dimension
        M = 2 # codimension
        A = numpy.array([[ 1./N +(n==m) for n in range(N)] for m in range(M)])
        def vector_f(x):
            return numpy.dot(A,x)

        x = numpy.array([1.*n for n in range(N)])
        ax = adouble(x)

        for r in range(R):
            trace_on(1)
            independent(ax)
            ay = vector_f(ax)
            dependent(ay)
            trace_off()
            u = numpy.random.rand(M)
            uJ = numpy.dot(u,A)
            assert_array_almost_equal( uJ, vec_jac(1,x,u, 0))

        for r in range(R):
            trace_on(r)
            independent(ax)
            ay = vector_f(ax)
            dependent(ay)
            trace_off()
            u = numpy.random.rand(M)
            uJ = numpy.dot(u,A)
            assert_array_almost_equal( uJ, vec_jac(r,x,u, 0))


    def test_hov_forward(self):
        """ checks only first order"""
        N = 3
        P = 1
        D = 1
        epsilon1 =  numpy.sqrt(10**-16)

        def f(x):
            return numpy.array([x[0]*x[1] + x[0]*x[2], x[1]*x[2]])

        x = numpy.array([1.,2.,3.])
        ax = adouble(x)
        trace_on(1)
        independent(ax)
        ay = f(ax)
        dependent(ay)
        trace_off()
        x = numpy.random.rand(N)
        V = numpy.random.rand(N,P,D)

        (y,W) = hov_forward(1, x, V)

        W2 = (f(x+epsilon1*V[:,0,0]) - f(x))/epsilon1
        W2 = W2.reshape((2,P,D))

        assert_array_almost_equal(y, f(x))
        assert_array_almost_equal(W, W2)

if __name__ == '__main__':
    try:
        import nose
    except:
        print 'Please install nose for unit testing'
    nose.runmodule()

