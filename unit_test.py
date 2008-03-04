#!/usr/bin/env python
import numpy as npy
from adolc import *

number_of_errors = 0
def near_equal(lhs,rhs):
	return ( abs((lhs-rhs)/rhs) < 10**(-10))

def near_equal_with_num_error_increase(lhs,rhs):
	global number_of_errors
	test_passed = near_equal( lhs.val, rhs)
	if test_passed is False:
		number_of_errors=number_of_errors+1
	return test_passed


def test_expression(str_expr, expr, ax, x):
	global number_of_errors
	print str_expr,'\t',expr(ax),'\t==',expr(x),'\t\t',
	test_passed = near_equal(expr(ax).val, expr(x))
	if test_passed is False:
		number_of_errors=number_of_errors+1
		print 'test failed!'
	else:
		print 'test OK!'

def test_expr(a_str_expr, str_expr):
	global number_of_errors
	exec('import xyz')

	
print 'testing constructors'
a = adouble(13.);	print a.__repr__(),'\t',;		print 'a=adouble(13.)\t= ',a,'\t\ta.val =',a.val
b = adouble(5);		print b.__repr__(),'\t',;		print 'b=adouble(5)\t= ',b,'\t\tb.val =',b.val
c = adouble(a);		print c.__repr__(),'\t',;		print 'c=adouble(a)\t= ',c,'\t\tc.val =',c.val

# unary
test_expression('-a:\t', lambda x: -x, a, a.val)
test_expression('-a:\t', lambda x: +x, a, a.val)

# operator + for int and double
test_expression('a + 2: ',	lambda x: x[0]+x[1], (a,2),		(a.val,2))
test_expression('a + 2.:',	lambda x: x[0]+x[1], (a,2.),	(a.val,2.))
test_expression('2 + a: ',	lambda x: x[0]+x[1], (2,a),		(2,a.val))
test_expression('2.+ a.:',	lambda x: x[0]+x[1], (2.,a),	(2., a.val))

# operator - for int and double
test_expression('a - 2: ',	lambda x: x[0]-x[1], (a,2),		(a.val,2))
test_expression('a - 2.:',	lambda x: x[0]-x[1], (a,2.),	(a.val,2.))
test_expression('2 - a: ',	lambda x: x[0]-x[1], (2,a),		(2,a.val))
test_expression('2.- a.:',	lambda x: x[0]-x[1], (2.,a),	(2., a.val))


# operator * for int and double
test_expression('a * 2: ',	lambda x: x[0]*x[1], (a,2),		(a.val,2))
test_expression('a * 2.:',	lambda x: x[0]*x[1], (a,2.),	(a.val,2.))
test_expression('2 * a: ',	lambda x: x[0]*x[1], (2,a),		(2,a.val))
test_expression('2.* a.:',	lambda x: x[0]*x[1], (2.,a),	(2., a.val))

# operator / for int and double
test_expression('a / 2: ',	lambda x: x[0]/x[1], (a,2),		(a.val,2))
test_expression('a / 2.:',	lambda x: x[0]/x[1], (a,2.),	(a.val,2.))
test_expression('2 / a: ',	lambda x: x[0]/x[1], (2,a),		(2,a.val))
test_expression('2./ a.:',	lambda x: x[0]/x[1], (2.,a),	(2., a.val))

#operator +,-,*,/ for badouble
test_expression('a + b: ',	lambda x: x[0]+x[1], (a,b),		(a.val,b.val))
test_expression('a - b: ',	lambda x: x[0]-x[1], (a,b),		(a.val,b.val))
test_expression('a * b: ',	lambda x: x[0]*x[1], (a,b),		(a.val,b.val))
test_expression('a / b: ',	lambda x: x[0]/x[1], (a,b),		(a.val,b.val))

# operator +=,-=,*=,/= for badouble
c = adouble(a)
d = c.val
c+=b; d+=b.val; print 'c+=b  \t ',c,'==',d,near_equal_with_num_error_increase(c,d)
c-=b; d-=b.val; print 'c-=b  \t ',c,'==',d,near_equal_with_num_error_increase(c,d)
c*=b; d*=b.val; print 'c*=b  \t ',c,'==',d,near_equal_with_num_error_increase(c,d)
c/=b; d/=b.val; print 'c/=b  \t ',c,'==',d,near_equal_with_num_error_increase(c,d)

# operator +=,-=,*=,/= for badouble
c = adouble(a)
d = c.val
c+=b.val; d+=b.val; print 'c+=b  \t ',c,'==',d,near_equal_with_num_error_increase(c,d)
c-=b.val; d-=b.val; print 'c-=b  \t ',c,'==',d,near_equal_with_num_error_increase(c,d)
c*=b.val; d*=b.val; print 'c*=b  \t ',c,'==',d,near_equal_with_num_error_increase(c,d)
c/=b.val; d/=b.val; print 'c/=b  \t ',c,'==',d,near_equal_with_num_error_increase(c,d)

# operator **
test_expression('a**2: ',	lambda x: x**2, a,		a.val)
print "to be implemented print '2.**a', 2.**a"


#functions
import numpy as npy
a = adouble(0.4);	print 'a=adouble(13.)\t= ',a,'\t\ta.val =',a.val
test_expression('exp  (a)     : ',		lambda x: npy.exp  (x),  a,		a.val)
test_expression('log  (a)     : ',		lambda x: npy.log  (x),  a,		a.val)
test_expression('sqrt (a)     : ',		lambda x: npy.sqrt (x),  a,		a.val)
test_expression('sin  (a)     : ',		lambda x: npy.sin  (x),  a,		a.val)
test_expression('cos  (a)     : ',		lambda x: npy.cos  (x),  a,		a.val)
test_expression('tan  (a)     : ',		lambda x: npy.tan  (x),  a,		a.val)
test_expression('asin (a)     : ',		lambda x: npy.arcsin (x),  a,		a.val)
test_expression('acos (a)     : ',		lambda x: npy.arccos (x),  a,		a.val)
test_expression('atan (a)     : ',		lambda x: npy.arctan (x),  a,		a.val)
test_expression('log10(a)     : ',		lambda x: npy.log10(x),  a,		a.val)
test_expression('sinh (a)     : ',		lambda x: npy.sinh (x),  a,		a.val)
test_expression('cosh (a)     : ',		lambda x: npy.cosh (x),  a,		a.val)
test_expression('tanh (a)     : ',		lambda x: npy.tanh (x),  a,		a.val)
test_expression('fabs (a)     : ',		lambda x: npy.fabs (x),  a,		a.val)
test_expression('ceil (a)     : ',		lambda x: npy.ceil (x),  a,		a.val)
test_expression('floor(a)    : ',		lambda x: npy.floor(x),	 a,		a.val)

#print 'exp  (a)', npy.exp  (a), a.exp  ()
#print 'log  (a)', npy.log  (a), a.log  ()
#print 'sqrt (a)', npy.sqrt (a), a.sqrt ()
#print 'sin  (a)', npy.sin  (a), a.sin  ()
#print 'cos  (a)', npy.cos  (a), a.cos  ()
#print 'tan  (a)', npy.tan  (a), a.tan  ()
#print 'asin (a)', npy.arcsin(a),a.asin ()
#print 'acos (a)', npy.arccos(a),a.acos ()
#print 'atan (a)', npy.arctan(a),a.atan ()
#print 'log10(a)', npy.log10(a), a.log10()
#print 'sinh (a)', npy.sinh (a), a.sinh ()
#print 'cosh (a)', npy.cosh (a), a.cosh ()
#print 'tanh (a)', npy.tanh (a), a.tanh ()
#print 'fabs (a)', npy.fabs (a), a.fabs ()
#print 'ceil (a)', npy.ceil (a), a.ceil ()
#print 'floor (a)',npy.floor (a),a.floor ()
print 'fmax (a,b)',			    a.fmax (b)
print 'fmax (a,0.3)',			a.fmax (0.3)
print 'fmax (0.3,a)','/* not implemented */'
print 'fmin (a,b)',			    a.fmin (b)
print 'fmin (a,0.3)',			a.fmin (0.3)
print 'fmin (0.3,a)','/* not implemented */'

N = 2
M = 3
P = N
D = 3
A = npy.zeros((M,N))
A[:] = [[ 1./N +(n==m) for n in range(N)] for m in range(M)]
x = npy.array([1./(i+1) for i in range(N)])
y = npy.zeros(M)
u = npy.zeros(M); u[0] = 1.
v = npy.zeros(N); v[0] = 1.
V = npy.array([[m==p for m in range(M)]for p in range(P)], dtype=float)
W = npy.array([[m==d for m in range(M)]for d in range(D)], dtype=float)

b = npy.zeros(N,dtype=float)
ax = npy.array([adouble(0.) for i in range(N)])

def scalar_f(x):
	global A
	return npy.dot(x,x)

def vector_f(x):
	global A
	return npy.dot(A,x)

trace_on(0)
for n in range(N):
	ax[n].is_independent(x[n])
ay = scalar_f(ax)
depends_on(ay)
trace_off()

trace_on(1)
for n in range(N):
	ax[n].is_independent(x[n])
ay = vector_f(ax)
for m in range(M):
	y[m] = depends_on(ay[m])
trace_off()

# basic drivers
print function(0,x)
print gradient(0,x)
print hessian(0,x)
print hess_vec(0,x,v)

print jacobian(1,x)
print vec_jac(1,x,u, 0)
print jac_vec(1,x,v)
print lagra_hess_vec(1,x,v,u)
#jac_solv(1,x,b, 0, 2); print b

# low level functions
print zos_forward(1,x,0)
print fos_forward(1,x,v,0)
print fov_forward(1,x,V)
print hos_forward(1,D,x,W,0)





