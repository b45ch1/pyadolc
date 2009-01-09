# This file is to be used with py.test

import sys
sys.path = ['.'] + sys.path #adding current working directory to the $PYTHONPATH
import numpy
import numpy.linalg
from adolc import *

def test_constructors():	
	a = adouble(13.);
	b = adouble(5)
	c = adouble(a)
	
	assert a.val == 13.
	assert b.val == 5
	assert c.val == 13.
	
def test_unary_operators():
	a = adouble(1.)
	b = -a
	assert b.val == -1.
	assert a.val == 1.
	
	print type(b)
	print type(a)
	
	
def test_radd():
	a = adouble(1.)
	b = a + 2.
	c = a + 2.
	d = 2.+ a
	
	assert a.val == 1.

def test_add():
	a = adouble(1.)
	b = a + 2.
	c = a + 2
	d = 2.+ a
	e = 2 + a
	
	assert b.val == 3.
	assert c.val == 3.
	assert d.val == 3.
	assert e.val == 3.
	
def test_sub():
	a = adouble(1.)
	b = a - 2.
	c = 2.- a
	
	assert b.val == -1.
	assert c.val == 1.
	
def test_mul():
	a = adouble(1.5)
	b = a * 2.
	c = 2.* a
	
	assert b.val == 3.
	assert c.val == 3.
	
def test_div():
	a = adouble(3.)
	b = a/2.
	c = 2./a
	
	assert b.val == 3./2.
	assert c.val == 2./3.

def test_independent():
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
	
def test_dependent():
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
	
	
def test_hov_wk_forward():
	def f(x):
		return numpy.sum(x)
	
	N = 10
	P = N
	D = 2
	keep = N+1
	
	x  = numpy.ones(N)
	ax = numpy.array([adouble(1.) for n in range(N)])
	
	trace_on(17)
	independent(ax)
	ay = f(ax)
	dependent(ay)
	trace_off()

	# directions V
	V = numpy.ones((N,P,D))
	(y,W) = hov_wk_forward(17, D, x, V, keep)
	
	print y
	print W
	
	assert False
	
	

## operator / for int and double
#test_expression('a / 2: ',	lambda x: x[0]/x[1], (a,2),		(a.val,2))
#test_expression('a / 2.:',	lambda x: x[0]/x[1], (a,2.),	(a.val,2.))
#test_expression('2 / a: ',	lambda x: x[0]/x[1], (2,a),		(2,a.val))
#test_expression('2./ a.:',	lambda x: x[0]/x[1], (2.,a),	(2., a.val))

##operator +,-,*,/ for badouble
#test_expression('a + b: ',	lambda x: x[0]+x[1], (a,b),		(a.val,b.val))
#test_expression('a - b: ',	lambda x: x[0]-x[1], (a,b),		(a.val,b.val))
#test_expression('a * b: ',	lambda x: x[0]*x[1], (a,b),		(a.val,b.val))
#test_expression('a / b: ',	lambda x: x[0]/x[1], (a,b),		(a.val,b.val))

## operator +=,-=,*=,/= for badouble
#c = adouble(a)
#d = c.val
#c+=b; d+=b.val; print 'c+=b  \t ',c,'==',d,near_equal_with_num_error_increase(c.val,d)
#c-=b; d-=b.val; print 'c-=b  \t ',c,'==',d,near_equal_with_num_error_increase(c.val,d)
#c*=b; d*=b.val; print 'c*=b  \t ',c,'==',d,near_equal_with_num_error_increase(c.val,d)
#c/=b; d/=b.val; print 'c/=b  \t ',c,'==',d,near_equal_with_num_error_increase(c.val,d)

## operator +=,-=,*=,/= for badouble
#c = adouble(a)
#d = c.val
#c+=b.val; d+=b.val; print 'c+=b  \t ',c,'==',d,near_equal_with_num_error_increase(c.val,d)
#c-=b.val; d-=b.val; print 'c-=b  \t ',c,'==',d,near_equal_with_num_error_increase(c.val,d)
#c*=b.val; d*=b.val; print 'c*=b  \t ',c,'==',d,near_equal_with_num_error_increase(c.val,d)
#c/=b.val; d/=b.val; print 'c/=b  \t ',c,'==',d,near_equal_with_num_error_increase(c.val,d)

## operator **
#test_expression('a**2: ',	lambda x: x**2, a,		a.val)
#print "to be implemented print '2.**a', 2.**a"


##functions
#import numpy as npy
#a = adouble(0.4);	print 'a=adouble(13.)\t= ',a,'\t\ta.val =',a.val
#test_expression('exp  (a)     : ',		lambda x: npy.exp  (x),  a,		a.val)
#test_expression('log  (a)     : ',		lambda x: npy.log  (x),  a,		a.val)
#test_expression('sqrt (a)     : ',		lambda x: npy.sqrt (x),  a,		a.val)
#test_expression('sin  (a)     : ',		lambda x: npy.sin  (x),  a,		a.val)
#test_expression('cos  (a)     : ',		lambda x: npy.cos  (x),  a,		a.val)
#test_expression('tan  (a)     : ',		lambda x: npy.tan  (x),  a,		a.val)
#test_expression('asin (a)     : ',		lambda x: npy.arcsin (x),  a,		a.val)
#test_expression('acos (a)     : ',		lambda x: npy.arccos (x),  a,		a.val)
#test_expression('atan (a)     : ',		lambda x: npy.arctan (x),  a,		a.val)
#test_expression('log10(a)     : ',		lambda x: npy.log10(x),  a,		a.val)
#test_expression('sinh (a)     : ',		lambda x: npy.sinh (x),  a,		a.val)
#test_expression('cosh (a)     : ',		lambda x: npy.cosh (x),  a,		a.val)
#test_expression('tanh (a)     : ',		lambda x: npy.tanh (x),  a,		a.val)
#test_expression('fabs (a)     : ',		lambda x: npy.fabs (x),  a,		a.val)
#test_expression('ceil (a)     : ',		lambda x: npy.ceil (x),  a,		a.val)
#test_expression('floor(a)    : ',		lambda x: npy.floor(x),	 a,		a.val)

##print 'exp  (a)', npy.exp  (a), a.exp  ()
##print 'log  (a)', npy.log  (a), a.log  ()
##print 'sqrt (a)', npy.sqrt (a), a.sqrt ()
##print 'sin  (a)', npy.sin  (a), a.sin  ()
##print 'cos  (a)', npy.cos  (a), a.cos  ()
##print 'tan  (a)', npy.tan  (a), a.tan  ()
##print 'asin (a)', npy.arcsin(a),a.asin ()
##print 'acos (a)', npy.arccos(a),a.acos ()
##print 'atan (a)', npy.arctan(a),a.atan ()
##print 'log10(a)', npy.log10(a), a.log10()
##print 'sinh (a)', npy.sinh (a), a.sinh ()
##print 'cosh (a)', npy.cosh (a), a.cosh ()
##print 'tanh (a)', npy.tanh (a), a.tanh ()
##print 'fabs (a)', npy.fabs (a), a.fabs ()
##print 'ceil (a)', npy.ceil (a), a.ceil ()
##print 'floor (a)',npy.floor (a),a.floor ()
#print 'fmax (a,b)',			    a.fmax (b)
#print 'fmax (a,0.3)',			a.fmax (0.3)
#print 'fmax (0.3,a)','/* not implemented */'
#print 'fmin (a,b)',			    a.fmin (b)
#print 'fmin (a,0.3)',			a.fmin (0.3)
#print 'fmin (0.3,a)','/* not implemented */'








#####################################################
## TESTING THE EVALUATION OF DERIVATIVES
#####################################################

#N = 6 # dimension
#M = 5 # codimension
#P = 4 # number of directional derivatives
#Q = 3 # number of adjoint derivatives
#D = 2 # order of derivatives

#A = npy.zeros((M,N))
#A[:] = [[ 1./N +(n==m) for n in range(N)] for m in range(M)]
#x = npy.array([1./(i+1) for i in range(N)])
#y = npy.zeros(M)
#u = npy.zeros(M); u[0] = 1.
#v = npy.zeros(N); v[0] = 1.
#Vnp = npy.array([[n==p for  p in range(P)]for n in range(N)], dtype=float)
#Vnd = npy.array([[n==d and d==0 for d in range(D)]for n in range(N)], dtype=float)
#Vnpd = npy.array([[[ n==p and d == 0 for d in range(D)] for p in range(P)] for n in range(N)], dtype = float)
#Uqm = npy.array([[q==n for m in range(M)]for q in range(Q)], dtype=float)

#b = npy.zeros(N,dtype=float)
#ax = npy.array([adouble(0.) for i in range(N)])

#def scalar_f(x):
	#global A
	#return npy.dot(x,x)

#def vector_f(x):
	#global A
	#return npy.dot(A,x)

#trace_on(0)
#for n in range(N):
	#ax[n].is_independent(x[n])
#ay = scalar_f(ax)
#depends_on(ay)
#trace_off()

#trace_on(1)
#for n in range(N):
	#ax[n].is_independent(x[n])
#ay = vector_f(ax)
#for m in range(M):
	#y[m] = depends_on(ay[m])
#trace_off()

## basic drivers
#print 'Function evaluation correct?\t\t',near_equal_with_num_error_increase(function(0,x), scalar_f(x))
#y = 2*x #gradient of scalar_f
#print 'Gradient evaluation correct?\t\t',near_equal_with_num_error_increase(gradient(0,x), y)
#H = 2*npy.eye(N) #hessian of scalar_f
#print 'Hessian evaluation correct?\t\t', near_equal_with_num_error_increase(hessian(0,x), H)
#Hv = npy.dot(H,v)
#print 'Hess_vec evaluation correct?\t\t', near_equal_with_num_error_increase(hess_vec(0,x,v), Hv)
#print 'Jacobian evaluation correct?\t\t', near_equal_with_num_error_increase(jacobian(1,x), A )
#uJ = npy.dot(u,A)
#print 'vec_jac evaluation correct?\t\t', near_equal_with_num_error_increase(vec_jac(1,x,u, 0), uJ )
#Jv = npy.dot(A,v)
#print 'vec_jac evaluation correct?\t\t', near_equal_with_num_error_increase(jac_vec(1,x,v), Jv )
#print 'lagra_hess_vec evaluation correct?\t', near_equal_with_num_error_increase(lagra_hess_vec(1,x,v,u), npy.zeros(N,dtype=float) )

##try:
	##jac_solv(1,x,b, 0, 2); print b
##except:
	##pass

## low level functions
#print 'zos_forward correct?\t\t\t', near_equal_with_num_error_increase(zos_forward(1,x,0), vector_f(x))
#print 'fos_forward correct?\t\t\t', near_equal_with_num_error_increase(fos_forward(1,x,v,0)[1], A[:,0])
#print 'fov_forward correct?\t\t\t', near_equal_with_num_error_increase(fov_forward(1,x,Vnp)[1], A[:,:P])
#print 'hov_forward correct?\t\t\t', near_equal_with_num_error_increase(hov_forward(1,D,x,Vnpd)[1][:,:,-1], npy.zeros((M,P)))
#uA = npy.dot(u,A)
#print 'fos_reverse correct?\t\t\t', near_equal_with_num_error_increase(fos_reverse(1,u), uA)
#UqmA = npy.dot(Uqm,A)
#print 'fov_reverse correct?\t\t\t', near_equal_with_num_error_increase(fov_reverse(1,Uqm), UqmA)
#print 'hos_forward correct?\t\t\t', near_equal_with_num_error_increase(hos_forward(1,D,x, Vnd,D+1)[1][:,-1], npy.zeros(M))
#print 'hos_reverse correct?\t\t\t', near_equal_with_num_error_increase(hos_reverse(1,D,u)[:,-1], npy.zeros(N))
#print 'hov_reverse correct?\t\t\t', near_equal_with_num_error_increase(hov_reverse(1,D,Uqm)[0][:,:,-1], npy.zeros((Q,N)))



## c style functions
#y = npy.zeros(1, dtype=float)
#g = npy.zeros(N, dtype=float)
#H = npy.zeros((N,N), dtype=float)
#z = npy.zeros(N, dtype=float)

#function(0,1,N,x,y)
#gradient(0,N,x,g)
#hessian(0,N,x,H)
#hess_vec(0, N, x,v, z)



#print 'number of failed tests =',number_of_errors



