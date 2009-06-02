# -*- coding: utf-8 -*-
# This file is to be used with py.test

import sys
sys.path = ['.'] + sys.path #adding current working directory to the $PYTHONPATH
import numpy
import numpy.linalg
import numpy.random
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal, assert_equal
from adolc import *

#def test_minimal_surface_objective_function_gradient():
	#"""
	#This test checks that the analytically computed gradient of a minimal surface problem objective function is
	#correctly computed by adolc
	#"""
	#def O_tilde(u):
		#""" this is the objective function"""
		#M = numpy.shape(u)[0]
		#h = 1./(M-1)
		#return M**2*h**2 + numpy.sum(0.25*( (u[1:,1:] - u[0:-1,0:-1])**2 + (u[1:,0:-1] - u[0:-1, 1:])**2))


	#def dO_tilde(u):
		#g = numpy.zeros(numpy.shape(u))
		#g[1:-1, 1:-1] = 2 * u[1:-1,1:-1] - 0.5*( u[0:-2,0:-2]  + u[2:,2:]  + u[:-2, 2:] + u [2:, :-2] )
		#return g


	## INITIAL VALUES
	#M = 5
	#h = 1./M
	#u = numpy.zeros((M,M),dtype=float)
	#u[0,:]=  [numpy.sin(numpy.pi*j*h/2.) for j in range(M)]
	#u[-1,:] = [ numpy.exp(numpy.pi/2) * numpy.sin(numpy.pi * j * h / 2.) for j in range(M)]
	#u[:,0]= 0
	#u[:,-1]= [ numpy.exp(i*h*numpy.pi/2.) for i in range(M)]


	#trace_on(1)
	#au = adouble(u)
	#independent(au)
	#ay = O_tilde(au)
	#dependent(ay)
	#trace_off()

	#ru = numpy.ravel(u)
	#rg = gradient(1,ru)
	#g = numpy.reshape(rg, numpy.shape(u))

	## on the edge the analytical solution is fixed to zero
	#g[:,0]  = 0
	#g[0,:]  = 0
	#g[:,-1] = 0
	#g[-1,:] = 0

	#assert_array_almost_equal(g, dO_tilde(u))

#def test_solve_minimal_surface_optimization_problem_with_projected_gradients():
	#"""
	#This is a minimal surface problem, discretized on a regular mesh with box constraints using projected gradients.
	#The necessary gradient is computed with adolc.
	#"""

	#def projected_gradients(x0, ffcn,dffcn, box_constraints, beta = 0.5, delta = 10**-3, epsilon = 10**-2, max_iter = 1000, line_search_max_iter = 100):
		#"""
		#INPUT:	box_constraints		[L,U], where L (resp. U) vector or matrix with the lower (resp. upper) bounds
		#"""
		#x = x0.copy()
		#L = numpy.array(box_constraints[0])
		#U = numpy.array(box_constraints[1])
		#def pgn(s):
			#a = 1.* (x>L)
			#b = 1.*(abs(x-L) <0.00001)
			#c = 1.*(s>0)
			#d = numpy.where( a + (b*c))
			#return numpy.sum(s[d])

		#def P(x, s, alpha):
			#x_alpha = x + alpha * s
			#a = x_alpha-L
			#b = U - x_alpha
			#return x_alpha - 1.*(a<0) * a + b * 1. * (b<0)

			
		#s = - dffcn(x)
		#k = 0
		#while pgn(s)>epsilon and k<= max_iter:
			#k +=1
			#s = - dffcn(x)
			#for m in range(line_search_max_iter):
				##print 'm=',m
				#alpha = beta**m
				#x_alpha = P(x,s,alpha)
				#if ffcn( x_alpha ) - ffcn(x) <= - delta * numpy.sum(s* (x_alpha - x)):
					#break
			#x_old = x.copy()
			#x = x_alpha

		#return x_old,s


	

	#def O_tilde(u):
		#""" this is the objective function"""
		#M = numpy.shape(u)[0]
		#h = 1./(M-1)
		#return M**2*h**2 + numpy.sum(0.25*( (u[1:,1:] - u[0:-1,0:-1])**2 + (u[1:,0:-1] - u[0:-1, 1:])**2))

	## INITIAL VALUES
	#M = 20
	#h = 1./M
	#u = numpy.zeros((M,M),dtype=float)
	#u[0,:]=  [numpy.sin(numpy.pi*j*h/2.) for j in range(M)]
	#u[-1,:] = [ numpy.exp(numpy.pi/2) * numpy.sin(numpy.pi * j * h / 2.) for j in range(M)]
	#u[:,0]= 0
	#u[:,-1]= [ numpy.exp(i*h*numpy.pi/2.) for i in range(M)]

	## tape the function evaluation
	#trace_on(1)
	#au = adouble(u)
	#independent(au)
	#ay = O_tilde(au)
	#dependent(ay)
	#trace_off()

	

	#def dO_tilde(u):
		#ru = numpy.ravel(u)
		#rg = gradient(1,ru)
		#g = numpy.reshape(rg, numpy.shape(u))
		
		## on the edge the analytical solution is fixed to zero
		#g[:,0]  = 0
		#g[0,:]  = 0
		#g[:,-1] = 0
		#g[-1,:] = 0
		
		#return g

	
	## X AND Y PARTITION
	#x_grid = numpy.linspace(0,1,M)
	#y_grid = numpy.linspace(0,1,M)

	## BOX CONSTRAINTS
	#lo = 2.5
	#L = numpy.zeros((M,M),dtype=float)

	#for n in range(M):
		#for m in range(M):
			#L[n,m] = 2.5 * ( (x_grid[n]-0.5)**2 + (y_grid[m]-0.5)**2 <= 1./16)

	#U = 100*numpy.ones((M,M),dtype=float)

	#Z,s = projected_gradients(u,O_tilde,dO_tilde,[L,U])


	## THIS CODE BELOW ONLY WORKS FOR MATPLOTLIB 0.91X
	#try:
		#import pylab
		#import matplotlib.axes3d as p3

		#x = y = range(numpy.shape(Z)[0])
		#X,Y = numpy.meshgrid(x_grid,y_grid)

		#fig=pylab.figure()
		#ax = p3.Axes3D(fig)
		#ax.plot_wireframe(X,Y,Z)

		#xs = Z + s
		#for n in range(M):
			#for m in range(M):
				#ax.plot3d([x_grid[m],x_grid[m]], [ y_grid[n], y_grid[n]], [Z[n,m], xs[n,m]], 'r')
		#ax.set_xlabel('X')
		#ax.set_ylabel('Y')
		#ax.set_zlabel('Z')
		#pylab.title('Minimal Surface')
		#pylab.savefig('./3D_plot.png')

		##pylab.show()
	#except:
		#pass

	### Plot with MAYAVI
	##try:
		##import enthought.mayavi.mlab as mlab
		##mlab.figure()
		##mlab.view(azimuth=130)
		##s = mlab.surf(x, y, Z, representation='wireframe', warp_scale='auto', line_width=1.)
		##mlab.savefig('./mayavi_3D_plot.png')
		###mlab.show()
	

	##except:
		##pass



#def test_chemical_reaction_equations():
	#""" Example of a chemical reaction. Provided by Tilman Barz (TU Berlin)"""

	#def g(k_ggw, sigma, nu, lam, q, c):
		#alpha  = k_ggw[:]*q[0] *c[0]
		#s = 0.
		#for i in range(3):
			#s += alpha[i] * (sigma[i] + nu[i]) * c[i] **(nu[i] - 1.)
		
		#return q[:] - lam * alpha[:] * c[:]/s

	#k_ggw = numpy.array([1., 1.45e-3, 3.5e-3])
	#sigma = numpy.array([0., 40., 40.])
	#nu    = numpy.array([1., 6.38, 5.14])
	#eps = 0.6
	#lam = 883.

	#c = numpy.array([118., 0.03, 0.03])
	#q = numpy.array([0.5, 0.5, 0.5])

	#ak_ggw = adouble(k_ggw)
	#asigma = adouble(sigma)
	#anu    = adouble(nu)
	#alam   = adouble(lam)
	#aq     = adouble(q)
	#ac     = adouble(c)

	#trace_on(1)
	#independent(ak_ggw)
	#independent(asigma)
	#independent(anu)
	#independent(alam)

	#ag = g(ak_ggw, asigma, anu, alam, aq, ac)

	#dependent(ag)
	#trace_off()

	#x = numpy.concatenate([k_ggw,sigma, nu, [lam]])
	#assert_array_almost_equal(g(k_ggw,sigma, nu, lam, q, c), function(1, x))


def test_ipopt_optimization():
	import pyipopt
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

	def eval_f_adolc(x, user_data = None):
		 return function(1,x)

	def eval_grad_f_adolc(x, user_data = None):
		 return gradient(1,x)

	def eval_g_adolc(x, user_data= None):
		return function(2,x)

	def eval_jac_g_adolc(x, flag, user_data = None):
		options = numpy.array([1,1,0,0],dtype=int)
		result = sparse.sparse_jac_no_repeat(2,x,options)
		if flag:
			return (result[1], result[2])
		else:
			return result[3]
	
	

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


	nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)
	start_time = time.time()
	x, zl, zu, obj = nlp.solve(x0)
	end_time = time.time()
	nlp.close()

	pure_python_optimization_time = end_time - start_time

	nlp_adolc = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f_adolc, eval_grad_f_adolc, eval_g_adolc, eval_jac_g_adolc, eval_h)
	start_time = time.time()
	x, zl, zu, obj = nlp_adolc.solve(x0)
	end_time = time.time()
	nlp_adolc.close()
	
	adolc_optimization_time = end_time - start_time
	print 'optimization time with derivatives computed by adolc = ', adolc_optimization_time
	print 'optimization time with derivatives computed by hand = ',pure_python_optimization_time
	assert adolc_optimization_time / pure_python_optimization_time < 10

	#print "Solution of the primal variables, x"
	#print x

	#print "Solution of the bound multipliers, z_L and z_U"
	#print zl, zu

	#print "Objective value"
	#print "f(x*) =", obj


	
	