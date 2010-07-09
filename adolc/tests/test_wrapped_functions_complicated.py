# -*- coding: utf-8 -*-
# This file is to be used with py.test

from numpy.testing import *
import numpy

import numpy.linalg
import numpy.random
import scipy.sparse
import unittest

import sys
sys.path = ['.'] + sys.path #adding current working directory to the $PYTHONPATH
from adolc import *
from adolc.cgraph import *


class TestMinimalSurfaceProblem(TestCase):
    def test_minimal_surface_objective_function_gradient(self):
        """
        This test checks that the analytically computed gradient of a minimal surface problem objective function is
        correctly computed by adolc
        """
        def O_tilde(u):
            """ this is the objective function"""
            M = numpy.shape(u)[0]
            h = 1./(M-1)
            return M**2*h**2 + numpy.sum(0.25*( (u[1:,1:] - u[0:-1,0:-1])**2 + (u[1:,0:-1] - u[0:-1, 1:])**2))
    
    
        def dO_tilde(u):
            g = numpy.zeros(numpy.shape(u))
            g[1:-1, 1:-1] = 2 * u[1:-1,1:-1] - 0.5*( u[0:-2,0:-2]  + u[2:,2:]  + u[:-2, 2:] + u [2:, :-2] )
            return g
    
    
        # INITIAL VALUES
        M = 5
        h = 1./M
        u = numpy.zeros((M,M),dtype=float)
        u[0,:]=  [numpy.sin(numpy.pi*j*h/2.) for j in range(M)]
        u[-1,:] = [ numpy.exp(numpy.pi/2) * numpy.sin(numpy.pi * j * h / 2.) for j in range(M)]
        u[:,0]= 0
        u[:,-1]= [ numpy.exp(i*h*numpy.pi/2.) for i in range(M)]
    
    
        trace_on(1)
        au = adouble(u)
        independent(au)
        ay = O_tilde(au)
        dependent(ay)
        trace_off()
    
        ru = numpy.ravel(u)
        rg = gradient(1,ru)
        g = numpy.reshape(rg, numpy.shape(u))
    
        # on the edge the analytical solution is fixed to zero
        g[:,0]  = 0
        g[0,:]  = 0
        g[:,-1] = 0
        g[-1,:] = 0
    
        assert_array_almost_equal(g, dO_tilde(u))
    
    def test_solve_minimal_surface_optimization_problem_with_projected_gradients(self):
        """
        This is a minimal surface problem, discretized on a regular mesh with box constraints using projected gradients.
        The necessary gradient is computed with adolc.
        """
    
        def projected_gradients(x0, ffcn,dffcn, box_constraints, beta = 0.5, delta = 10**-3, epsilon = 10**-2, max_iter = 1000, line_search_max_iter = 100):
            """
            INPUT:	box_constraints		[L,U], where L (resp. U) vector or matrix with the lower (resp. upper) bounds
            """
            x = x0.copy()
            L = numpy.array(box_constraints[0])
            U = numpy.array(box_constraints[1])
            def pgn(s):
                a = 1.* (x>L)
                b = 1.*(abs(x-L) <0.00001)
                c = 1.*(s>0)
                d = numpy.where( a + (b*c))
                return numpy.sum(s[d])
    
            def P(x, s, alpha):
                x_alpha = x + alpha * s
                a = x_alpha-L
                b = U - x_alpha
                return x_alpha - 1.*(a<0) * a + b * 1. * (b<0)
    
                
            s = - dffcn(x)
            k = 0
            while pgn(s)>epsilon and k<= max_iter:
                k +=1
                s = - dffcn(x)
                for m in range(line_search_max_iter):
                    #print 'm=',m
                    alpha = beta**m
                    x_alpha = P(x,s,alpha)
                    if ffcn( x_alpha ) - ffcn(x) <= - delta * numpy.sum(s* (x_alpha - x)):
                        break
                x_old = x.copy()
                x = x_alpha
    
            return x_old,s
    
    
        
    
        def O_tilde(u):
            """ this is the objective function"""
            M = numpy.shape(u)[0]
            h = 1./(M-1)
            return M**2*h**2 + numpy.sum(0.25*( (u[1:,1:] - u[0:-1,0:-1])**2 + (u[1:,0:-1] - u[0:-1, 1:])**2))
    
        # INITIAL VALUES
        M = 20
        h = 1./M
        u = numpy.zeros((M,M),dtype=float)
        u[0,:]=  [numpy.sin(numpy.pi*j*h/2.) for j in range(M)]
        u[-1,:] = [ numpy.exp(numpy.pi/2) * numpy.sin(numpy.pi * j * h / 2.) for j in range(M)]
        u[:,0]= 0
        u[:,-1]= [ numpy.exp(i*h*numpy.pi/2.) for i in range(M)]
    
        # tape the function evaluation
        trace_on(1)
        au = adouble(u)
        independent(au)
        ay = O_tilde(au)
        dependent(ay)
        trace_off()
    
    
        def dO_tilde(u):
            ru = numpy.ravel(u)
            rg = gradient(1,ru)
            g = numpy.reshape(rg, numpy.shape(u))
            
            # on the edge the analytical solution is fixed to zero
            g[:,0]  = 0
            g[0,:]  = 0
            g[:,-1] = 0
            g[-1,:] = 0
            
            return g
    
        
        # X AND Y PARTITION
        x_grid = numpy.linspace(0,1,M)
        y_grid = numpy.linspace(0,1,M)
    
        # BOX CONSTRAINTS
        lo = 2.5
        L = numpy.zeros((M,M),dtype=float)
    
        for n in range(M):
            for m in range(M):
                L[n,m] = 2.5 * ( (x_grid[n]-0.5)**2 + (y_grid[m]-0.5)**2 <= 1./16)
    
        U = 100*numpy.ones((M,M),dtype=float)
    
        Z,s = projected_gradients(u,O_tilde,dO_tilde,[L,U])
    
    
        # THIS CODE BELOW ONLY WORKS FOR MATPLOTLIB 0.91X
        try:
            import pylab
            import matplotlib.axes3d as p3
    
            x = y = range(numpy.shape(Z)[0])
            X,Y = numpy.meshgrid(x_grid,y_grid)
    
            fig=pylab.figure()
            ax = p3.Axes3D(fig)
            ax.plot_wireframe(X,Y,Z)
    
            xs = Z + s
            for n in range(M):
                for m in range(M):
                    ax.plot3d([x_grid[m],x_grid[m]], [ y_grid[n], y_grid[n]], [Z[n,m], xs[n,m]], 'r')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            pylab.title('Minimal Surface')
            pylab.savefig('./3D_plot.png')
            pylab.savefig('./3D_plot.eps')
    
            #pylab.show()
        except:
            print '3d plotting with matplotlib failed'
            pass
    
        # Plot with MAYAVI
        try:
            import enthought.mayavi.mlab as mlab
            mlab.figure()
            mlab.view(azimuth=130)
            s = mlab.surf(x, y, Z, representation='wireframe', warp_scale='auto', line_width=1.)
            mlab.savefig('./mayavi_3D_plot.png')
            #mlab.show()
    
        except:
            pass


class TestChemicalReaction(TestCase):
    def test_chemical_reaction_equations(self):
        """ Example of a chemical equilibrium. Provided by Tilman Barz (TU Berlin)"""
    
        def g(k_ggw, sigma, nu, lam, q, c):
            alpha = k_ggw[:]*q[0] *c[0]**(nu[:] - 1.)
            s = 0.
            for i in range(3):
                s += alpha[i] * (sigma[i] + nu[i]) * c[i]
            
            return q[:] - lam * alpha[:] * c[:]/s
    
        k_ggw = numpy.array([1., 1.45e-3, 3.5e-3])
        sigma = numpy.array([0., 40., 40.])
        nu    = numpy.array([1., 6.38, 5.14])
        eps = 0.6
        lam = 883.
    
        c = numpy.array([118., 0.03, 0.03])
        q = numpy.array([0.5, 0.5, 0.5])
    
        ak_ggw = adouble(k_ggw)
        asigma = adouble(sigma)
        anu    = adouble(nu)
        alam   = adouble(lam)
        aq     = adouble(q)
        ac     = adouble(c)
    
        trace_on(1)
        independent(ak_ggw)
        independent(asigma)
        independent(anu)
        independent(alam)
    
        ag = g(ak_ggw, asigma, anu, alam, aq, ac)
    
        dependent(ag)
        trace_off()
    
        x = numpy.concatenate([k_ggw,sigma, nu, [lam]])
        assert_array_almost_equal(g(k_ggw,sigma, nu, lam, q, c), function(1, x))

class TestNonlinearRegression(TestCase):
    
    def test_nonlinear_regression(self):
        """
        Original code from the Scipy Mailing List posted by Sturla Molden

        Goal: Regression of a nonlinear model
            y = Vmax * x / (x + Km)
            
        Check here that the leastsq function working with the  ADOLC computed jacobian gives the correct result.
        """
        
        import numpy as np
        import scipy
        from scipy.linalg import qr, solve
        from scipy.optimize import leastsq
        
        # Bates and Watts (1988) puromycin data

        data = ((0.02, 47, 76),
               (0.06, 97, 107),
               (0.11, 123, 139),
               (0.22, 152, 159),
               (0.56, 191, 201),
               (1.10, 200, 207))
        
        data = np.array(data)
        
        X = data[:,0:1].repeat(2,axis=1).flatten()
        Y = data[:,1:].flatten()
        
        # initial fit from Linewaver-Burk plot
        y = Y**-1
        x = np.vstack((np.ones(X.shape),X**-1)).T
        q,r = qr(x, econ=True)
        b = solve(r, (np.mat(y) * np.mat(q)).T).ravel()
        
        # Michaelis-Menten fit from Lineweaver-Burk
        Vmax = 1.0 / b[0]
        Km = b[1] * Vmax
        
        # refit with Levenberg-Marquardt method
        
        def michaelis_menten(t, x):
           Vmax, Km = t
           return Vmax*x/(x + Km)
        
        def residuals(t, x, y):
           return y - michaelis_menten(t, x)
        
        AP = AdolcProgram()
        def trace_residuals(t, x, y):
            AP.trace_on(1)
            at = adouble(t)
            # ax = adouble(x)
            # ay = adouble(y)
            AP.independent(at)
            # AP.independent(ax)
            # AP.independent(ay)
            ar = residuals(at,x,y)
            AP.dependent(ar)
            AP.trace_off()
            
        def adolc_Jacobian(t,x,y):
            return AP.jacobian([t])
            
        trace_residuals([Vmax,Km],X,Y)
        
        # print adolc_Jacobian([Vmax,Km],X,Y)
        (Vmax1,Km1),ierr1 = leastsq(residuals, (Vmax,Km), args=(X,Y))

        (Vmax2,Km2),ierr2 = leastsq(residuals, (Vmax,Km), args=(X,Y),
        Dfun=adolc_Jacobian, col_deriv=0)
        
        assert_array_almost_equal([Vmax1,Km1],[Vmax2,Km2])
        
        
        
        
    
    
    
    def test_model_fit_example_from_scipy_mailing_list(self):
        """
        provided by Ernest Adrogué <eadrogue@gmx.net>
        modified to use numpy package instead of the math package
    
        also: numpy.abs doesn't work correctly yet, using numpy.fabs
        
        """
    
        import random
    
        class Observation(object):
            def __init__(self, data):
                self.ht = data[0]
                self.at = data[1]
                self.hg = data[2]
                self.ag = data[3]
            def __repr__(self):
                return "[%s %s %s %s]"%(self.ht, self.at, self.hg, self.ag)
    
        def random_obs():
            names = 'abcdefghijklmnopqrstuvwxyz'
            random_data = [(i,j,random.randint(0,5),random.randint(0,5))
                            for i in names for j in names if i != j]
            return [Observation(i) for i in random_data]
    
        class Model(object):
            def __init__(self, observations):
                self.observations = tuple(observations)
                self.names = [i.ht for i in observations]
                self.names.extend([i.at for i in observations])
                self.names = tuple(set(self.names))
                self.n = len(self.names)
                
            def guess(self):
                a, b = [], []
                total_hg = numpy.sum([i.hg for i in self.observations])
                for j in self.names:
                    sh_j = numpy.sum([i.hg for i in self.observations if i.ht == j])
                    ca_j = numpy.sum([i.hg for i in self.observations if i.at == j])
                    a.append(sh_j/numpy.sqrt(total_hg))
                    b.append(ca_j/numpy.sqrt(total_hg))
                    
                return a + b + [1.5, 0]
                
            def tau(self, mu1, mu2, rho, x, y):
                if x == 0 and y == 0:
                    t = 1 - mu1 * mu2 * rho
                elif x == 0 and y == 1:
                    t = 1 + mu1 * rho
                elif x == 1 and y == 0:
                    t = 1 + mu2 * rho
                elif x == 1 and y == 1:
                    t = 1 - rho
                else:
                    t = 1
                return t
                
            def fobj(self, x):
                n = self.n
                y = [numpy.fabs(i) for i in x[:-2]]
                y.insert(0, numpy.fabs(n - sum(y[:n-1])))
                a = dict(zip(self.names, y[:n]))
                b = dict(zip(self.names, y[n:]))
                g = numpy.fabs(x[-2])
                r = x[-1]
                pseudo_loglikelihood = 0
                for m in self.observations:
                    x = m.hg
                    y = m.ag
                    mu1 = a[m.ht] * b[m.at] * g
                    mu2 = a[m.at] * b[m.ht]
                    tau = self.tau(mu1, mu2, r, m.hg, m.ag)
                    mu1 = mu1 > 0 and mu1 or 1e-10
                    mu2 = mu2 > 0 and mu2 or 1e-10
                    tau = tau > 0 and tau or 1e-10
                    pseudo_loglikelihood += numpy.log(tau)
                    pseudo_loglikelihood += m.hg * numpy.log(mu1) - mu1
                    pseudo_loglikelihood += m.ag * numpy.log(mu2) - mu2
                return -pseudo_loglikelihood
    
    
            def tape_fobj(self):
                x = self.guess()
                ax = adouble(x)
                trace_on(1)
                independent(ax)
                ay = self.fobj(ax)
                dependent(ay)
                trace_off()
    
            def fobj_adolc(self,x):
                return function(1,x)
    
            def gobj_adolc(self,x):
                return gradient(1,x)
    
            def gobj_finite_differences(self,x):
                epsilon = 10**-8
                N = numpy.size(x)
                assert numpy.ndim(x) == 1
    
                I = numpy.eye(N)
    
                retval = numpy.zeros(N,dtype=float)
    
                for n in range(N):
                    retval[n] = (self.fobj(x + epsilon*I[:,n])  - self.fobj(x))/epsilon
    
                return retval
                
    
        m = Model(random_obs())
        x0 = m.guess()
        m.tape_fobj()
        assert_almost_equal(m.fobj(x0), m.fobj_adolc(x0)[0])
        assert_array_almost_equal(m.gobj_finite_differences(x0), m.gobj_adolc(x0), decimal = 3)


class TestDifferentiatedIntegrators(TestCase):

    def Runge_Kutta_step_to_test_hov_forward(self):
        # defining the butcher tableau
        c =      numpy.array([0., 1./4., 3./8., 12./13., 1., 1./2. ], dtype=float)
        b =      numpy.array([16./135., 0., 6656./12825., 28561./56430., -9./50., 2./55.], dtype=float)
        A      = numpy.array([[0.,0.,0.,0.,0.,0.],
                            [1./4., 0., 0., 0., 0., 0.],
                            [3./32., 9./32., 0., 0., 0., 0.],
                            [1932./2197., -7200./2197., 7296./2197., 0., 0., 0.],
                            [439./216., -8., 3680./513., -845./4104., 0., 0.],
                            [-8./27., 2., -3544./2565., 1859./4104., -11./40., 0.]]
                            , dtype=float)
            
        # define the rhs of the ODE
        def f(t,x,p):
            return numpy.array([x[1], -2.*p[0]*x[1] - p[1]*p[1]*x[0]])
    
        # set the initial values
        t = 0.
        h = 0.01
        x = numpy.array([1.,0.], dtype=float)
        p  = numpy.array([0.1, 3.]) # p=(r,omega)
        V = numpy.zeros((4, 2, 1), dtype=float)
        V[2:,:,0] = numpy.eye(2, dtype=float)
        N = numpy.size(x)
        S = numpy.size(b)
    
        
    
        # tape Runge Kutta step
        trace_on(1)
        at = adouble(t)
        ah = adouble(h)
        ax = adouble(x)
        ap = adouble(p)
        ak = adouble(numpy.zeros((S, N)))
        
        independent(ax)
        independent(ap)
        
        for s in range(S):
            ak[s,:] = f(at + ah * c, ax + ah * numpy.dot(A[s,:], ak), ap)
        ay = ax[:] +  ah * numpy.dot(b, ak)
    
        dependent(ay)
        trace_off()
    
        z = numpy.concatenate([x,p])
        (y,W) = hov_forward(1, z, V)
    
        def phi(t,r,w,x):
            return x*(w**2 - r**2)**(-1./2)*(r*numpy.sin(t*(w**2 - r**2)**(1./2)) + (w**2 - r**2)**(1./2)*numpy.cos(t*(w**2 - r**2)**(1./2)))*numpy.exp(-r*t)
    
        def dphidr(t,r,w,x):
            return x*(w**2 - r**2)**(-1./2.)*(r*t*numpy.sin(t*(w**2 - r**2)**(1./2.)) - r*(w**2 - r**2)**(-1./2.)*numpy.cos(t*(w**2 - r**2)**(1./2.)) - t*r**2*(w**2 - r**2)**(-1./2.)*numpy.cos(t*(w**2 - r**2)**(1./2.)) + numpy.sin(t*(w**2 - r**2)**(1./2.)))*numpy.exp(-r*t) + r*x*(w**2 - r**2)**(-3./2.)*(r*numpy.sin(t*(w**2 - r**2)**(1./2.)) + (w**2 - r**2)**(1./2.)*numpy.cos(t*(w**2 - r**2)**(1./2.)))*numpy.exp(-r*t) - t*x*(w**2 - r**2)**(-1./2.)*(r*numpy.sin(t*(w**2 - r**2)**(1./2.)) + (w**2 - r**2)**(1./2.)*numpy.cos(t*(w**2 - r**2)**(1./2.)))*numpy.exp(-r*t)
            
    
        def dphidw(t,r,w,x):
            return x*(w**2 - r**2)**(-1./2.)*(w*(w**2 - r**2)**(-1./2.)*numpy.cos(t*(w**2 - r**2)**(1./2.)) - t*w*numpy.sin(t*(w**2 - r**2)**(1./2.)) + r*t*w*(w**2 - r**2)**(-1./2.)*numpy.cos(t*(w**2 - r**2)**(1./2.)))*numpy.exp(-r*t) - w*x*(w**2 - r**2)**(-3./2.)*(r*numpy.sin(t*(w**2 - r**2)**(1./2.)) + (w**2 - r**2)**(1./2.)*numpy.cos(t*(w**2 - r**2)**(1./2.)))*numpy.exp(-r*t)
    
        y_exact = phi(h,p[0],p[1],x[0])
        W_exact = numpy.array([  dphidr(h,p[0],p[1],x[0]),  dphidw(h,p[0],p[1],x[0]) ])
    
        assert_almost_equal(W[0,:,0], W_exact)
        assert_almost_equal(y[0],y_exact)


        # plot to check the analytical solution
        #eps = 10**-16
        #epsilon = numpy.sqrt(eps)
    
        #import pylab
        #pylab.figure()
        #ts = numpy.linspace(0,50,1000)
        #phis    = phi(ts,p[0],p[1],x[0])
        #dphidrs = dphidr(ts,p[0],p[1],x[0])
        #dphidws = dphidw(ts,p[0],p[1],x[0])
        #dphidrsfd = (phi(ts,p[0] + epsilon,p[1],x[0]) - phi(ts,p[0],p[1],x[0]))/epsilon
        #dphidwsfd = (phi(ts,p[0],p[1] + epsilon,x[0]) - phi(ts,p[0],p[1],x[0]))/epsilon
        #pylab.plot(ts, phis, 'b-', label=r'$x(t)$')
        #pylab.plot(ts,dphidrs,'r-', label = r' $\frac{d x}{d r}(t)$' )
        #pylab.plot(ts,dphidws,'g-', label = r' $\frac{d x}{d w}(t)$' )
        #pylab.plot(ts,dphidrsfd,'r.', label = r'FD: $\frac{d x}{d r}(t)$' )
        #pylab.plot(ts,dphidwsfd,'g.', label = r'FD: $\frac{d x}{d w}(t)$' )
    
        #pylab.legend()
        #pylab.show()
        ##assert False







if __name__ == "__main__":
    run_module_suite()