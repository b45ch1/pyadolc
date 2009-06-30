import numpy
import _adolc

try:
	import sparse
except:
	print 'Notice: sparse drivers not available'
	

from _adolc import *

__doc__ = """
 Adolc: Algorithmic Differentiation Software 
	see http://www.math.tu-dresden.de/~adol-c/ for documentation of Adolc 
	http://github.com/b45ch1/pyadolc/tree/master for more information and documentation of this Python extension
	
	return values are always numpy arrays!
	
	Example Session: 
	from numpy import *
	from adolc import * 
	def vector_f(x): 
	\tV=vander(x) 
	\treturn dot(v,x)

	x = arange(5,dtype=float)
	ax = adouble(x)
	
	trace_on(0)
	independent(ax)
	ay = vector_f(ax)
	dependent(ay)
	trace_off()

	x = array([1.,4.,0.,0.,0.])
	y = function(0,x)
	J = jacobian(0,x)
"""

def adouble(x):
	"""
	Return adouble from scalar or array of arbitrary shape
	INPUT:  either a float or an array of floats
	OUTPUT: adouble or array of adoubles
	"""
	if numpy.isscalar(x):
		return _adolc.adouble(float(x))
	elif isinstance(x,_adolc.adouble):
		return _adolc.adouble(x)
	else:
		x = numpy.asarray(x, dtype=float)
		shp = numpy.shape(x)
		xr = numpy.ravel(x)
		axr = numpy.array([_adolc.adouble(xr[n]) for n in range(len(xr))])
		ax = axr.reshape(shp)
		return ax

def adub(x):
	""" Dummy function to prevent users using adubs directly"""
	raise NotImplementedError("Warning: Do not use adub directly. May result in incorrect computations!")


def iadouble(x):
	"""
	INPUT :  numpy.array x of type float
	OUTPUT:  numpy.array ax of type adouble of the same dimension as x
	Example usage:
	ax = iadouble(x)

	is equivalent to:
	ax = adouble(0.)
	ax.is_independent(x)
	
	"""

	ax = adouble(0.)
	ax.is_independent(x)
	return ax

def independent(ax):
	"""
	independent(ax)
	INPUT :  numpy.array of type adouble
	OUTPUT:  ax
	Mark ax as independent
	"""

	if isinstance(ax, _adolc.adouble):
		x = ax.val
		ax.is_independent(x)
		return ax
	else:
		shp = numpy.shape(ax)
		axr = numpy.ravel(ax)
		N   = numpy.size(axr)
		xr = numpy.array([axr[n].val for n in range(N)])
		map(_adolc.adouble.is_independent,axr,xr)
		return ax
	

def dependent(ax):
	"""
	dependent(ax)
	INPUT :  numpy.array of type adouble
	OUTPUT:  ax
	
	Mark ax as dependent.
	"""
	if isinstance(ax, _adolc.adouble):
		depends_on(ax)
		return ax
	else:
		axr = numpy.ravel(ax)
		N   = numpy.size(axr)
		
		for n in range(N):
			if numpy.isscalar(axr[n]):
				axr[n] = adouble(axr[n])
			depends_on(axr[n])
		return ax


def trace_on(tape_tag):
	"""
	start recording to the tape with index tape_tag
	"""
	assert type(tape_tag) == int
	return _adolc.trace_on(tape_tag)

def trace_of(tape_tag):
	"""turn off tracing"""
	assert type(tape_tag) == int
	return _adolc.trace_off()

def function(tape_tag,x):
	"""
	evaluate the function f(x) recorded on tape with index tape_tag
	"""
	assert type(tape_tag) == int
	x = numpy.asarray(x, dtype=float)
	return _adolc.function(tape_tag,x)

def gradient(tape_tag,x):
	"""
	evaluate the gradient g = f'(x), f:R^N -> R
	"""
	assert type(tape_tag) == int
	x = numpy.asarray(x, dtype=float)
	return _adolc.gradient(tape_tag,x)

def hessian(tape_tag,x):
	"""
	evaluate the hessian H = f\"(x), f:R^N -> R"
	"""
	assert type(tape_tag) == int
	x = numpy.asarray(x, dtype=float)
	return _adolc.hessian(tape_tag,x)

def jacobian(tape_tag,x):
	"""
	evaluate the jacobian J = F'(x), F:R^N -> R^M
	"""
	assert type(tape_tag) == int
	x = numpy.asarray(x, dtype=float)
	return _adolc.jacobian(tape_tag,x)

def vec_jac(tape_tag,x,u, repeat):
	"""
	evaluate u^T F'(x), F:R^N -> R^M
	"""
	assert type(repeat) == bool or type(repeat) == int
	assert type(tape_tag) == int
	x = numpy.asarray(x, dtype=float)
	u = numpy.asarray(u,dtype=float)
	return _adolc.vec_jac(tape_tag,x,u, repeat)

def jac_vec(tape_tag,x,v):
	"""
	evaluate  F'(x)v, F:R^N -> R^M
	"""
	assert type(tape_tag) == int
	x = numpy.asarray(x, dtype=float)
	v = numpy.asarray(v, dtype=float)
	return _adolc.jac_vec(tape_tag,x,v)

def hess_vec(tape_tag,x,v):
	"""
	evaluate  f''(x)v, f:R^N -> R
	"""
	assert type(tape_tag) == int
	assert numpy.ndim(x)  == 1
	assert numpy.ndim(v)  == 1
	assert numpy.size(x)  == numpy.size(v)
	
	x = numpy.asarray(x, dtype=float)
	v = numpy.asarray(v, dtype=float)
	return _adolc.hess_vec(tape_tag,x,v)


def lagra_hess_vec(tape_tag,x,u,v):
	"""
	evaluate  u^T F''(x)v, F:R^N -> R^M
	"""
	assert type(tape_tag) == int
	assert numpy.ndim(x)  == 1
	assert numpy.ndim(u)  == 1
	assert numpy.ndim(v)  == 1
	assert numpy.size(x)  == numpy.size(v)
	
	x = numpy.asarray(x, dtype=float)
	u = numpy.asarray(u, dtype=float)
	v = numpy.asarray(v, dtype=float)
	return _adolc.lagra_hess_vec(tape_tag,x,u,v)

def zos_forward(tape_tag,x,keep):
	"""
	zero order scalar forward:
	y = zos_forward(tape_tag,x,keep)
	F:R^N -> R^M
	x is N-vector, y is M-vector
	keep = 1 prepares for fos_reverse or fov_reverse
	"""
	assert type(tape_tag) == int
	assert type(keep) == int
	assert numpy.ndim(x)  == 1
	
	x = numpy.asarray(x, dtype=float)
	return _adolc.zos_forward(tape_tag, x, keep)

def fos_forward(tape_tag, x, v, keep):
	"""
	first order scalar forward:
	(y,w) = fos_forward(tape_tag, x, v, keep)
	F:R^N -> R^M
	x is N-array, y is M-array
	v is N-array, direction
	w is M-array, directional derivative
	keep = 1 prepares for fos_reverse or fov_reverse
	keep = 2 prepares for hos_reverse or hov_reverse
	"""
	assert type(tape_tag) == int
	assert type(keep) == int
	assert numpy.ndim(x)  == 1
	assert numpy.ndim(v)  == 1
	assert numpy.size(x)  == numpy.size(v)
	
	x = numpy.asarray(x, dtype=float)
	v = numpy.asarray(v, dtype=float)
	return _adolc.fos_forward(tape_tag,x,v,keep)

def fov_forward(tape_tag,x,V):
	"""
	first order vector forward:
	(y,W) = fov_forward(tape_tag, x, V)
	F:R^N -> R^M
	x is N-vector, y is M-vector
	V is (N x P)-matrix. P directions
	W is (M x P)-matrix. P directiona derivatives	
	"""
	assert type(tape_tag) == int
	assert numpy.ndim(V) == 2
	assert numpy.ndim(x)  == 1
	assert numpy.size(x)  == numpy.shape(V)[0]
	
	x = numpy.asarray(x, dtype=float)
	V = numpy.asarray(V, dtype=float)
	return _adolc.fov_forward(tape_tag,x,V)

def hos_forward(tape_tag, x, V, keep):
	"""
	higher order scalar forward:
	(y,W) = hos_forward(tape_tag, x, V, keep)
	F:R^N -> R^M
	x is N-vector, y is M-vector
	D is the highest order of the derivative
	V is (N x D)-matrix
	W is (M x D)-matrix
	keep = 1 prepares for fos_reverse or fov_reverse
	D+1 >= keep > 2 prepares for hos_reverse or hov_reverse
	"""
	assert type(tape_tag) == int
	assert type(keep)     == int
	assert numpy.ndim(V)  == 2
	assert numpy.ndim(x)  == 1
	assert numpy.size(x)  == numpy.shape(V)[0]
	
	x = numpy.asarray(x, dtype=float)
	V = numpy.asarray(V, dtype=float)
	return _adolc.hos_forward(tape_tag, x, V, keep)

def hov_forward(tape_tag, x, V):
	"""
	higher order vector forward:
	(y,W) = hov_forward(tape_tag, x, V)
	F:R^N -> R^M
	x is N-vector, y is M-vector
	D is the order of the derivative
	V is (N x P x D)-matrix, P directions
	W is (M x P x D)-matrix, P directional derivatives
	

	"""
	assert type(tape_tag) == int
	assert numpy.ndim(V) == 3
	assert numpy.ndim(x) == 1
	assert numpy.size(x) == numpy.shape(V)[0]
	x = numpy.asarray(x, dtype=float)
	V = numpy.asarray(V, dtype=float)
	return _adolc.hov_forward(tape_tag, x, V)


def hov_wk_forward(tape_tag, x, V, keep):
	"""
	higher order vector forward with keep:
	(y,W) = hov_wk_forward(tape_tag, x, V, keep)
	F:R^N -> R^M
	x is N-vector, y is M-vector
	D is the order of the derivative
	V is (N x P x D)-matrix, P directions
	W is (M x P x D)-matrix, P directional derivatives
	"""
	assert type(tape_tag) == int
	assert type(keep) == int
	assert numpy.ndim(V) == 3
	assert numpy.ndim(x) == 1
	assert numpy.size(x) == numpy.shape(V)[0]
	x = numpy.asarray(x, dtype=float)
	V = numpy.asarray(V, dtype=float)
	return _adolc.hov_wk_forward(tape_tag, x, V, keep)


def fos_reverse(tape_tag, u):
	"""
	first order scalar reverse:
	z = fos_reverse(tape_tag, u)
	F:R^N -> R^M
	u is M-vector, adjoint direction
	z is N-vector, adjoint directional derivative z= u F'(x)
	after calling zos_forward, fos_forward or hos_forward with keep = 1
													

	"""
	assert type(tape_tag) == int
	assert numpy.ndim(u) == 1
	u = numpy.asarray(u,dtype=float)
	return _adolc.fos_reverse(tape_tag, u)


def fov_reverse(tape_tag, U):
	"""
	first order vector reverse:
	Z = fov_reverse(tape_tag,  U)
	F:R^N -> R^M
	U is (QxM)-matrix, Q adjoint directions
	Z is (QxN)-matrix, adjoint directional derivative Z = U F'(x)
	after calling zos_forward, fos_forward or hos_forward with keep = 1
	
	"""
	assert type(tape_tag) == int
	assert numpy.ndim(U) == 2
	U = numpy.asarray(U, dtype=float)
	return _adolc.fov_reverse(tape_tag, U)


def hos_reverse(tape_tag, D, u):
	"""
	higher order scalar reverse:
	Z = hos_reverse(tape_tag, D, u)
	F:R^N -> R^M
	D is the order of the derivative
	u is M-vector, adjoint vector
	Z is (N x D+1)-matrix, adjoint directional derivative Z = [u^T F'(x), u^T F\" v[:,0],U F\" v[:,1] + 0.5 u^T F^(3) v[:,0],...]
	after calling fos_forward or hos_forward with keep = D+1
													
	"""
	assert type(tape_tag) == int
	assert type(D) == int
	assert numpy.ndim(u) == 1
	u = numpy.asarray(u, dtype=float)
	return _adolc.hos_reverse(tape_tag, D, u)


def hov_reverse(tape_tag, D, U):
	"""
,	higher order vector reverse:
	(Z,nz) = hov_reverse(tape_tag, D, U)
	F:R^N -> R^M
	D is the order of the derivative
	U is (Q x M)-matrix, Q adjoint directions
	Z is (Q x N x D+1)-matrix, adjoint directional derivative Z = [U F'(x), U F\" v[:,0],  U F\" v[:,1] + 0.5 U F^(3) v[:,0],... ]
	nz is (Q x N)-matrix, information about the sparsity of Z:
	0:trivial, 1:linear, 2:polynomial, 3:rational, 4:transcendental, 5:non-smooth
	after calling fos_forward or hos_forward with keep = D+1
	"""
	assert type(tape_tag) == int
	assert type(D) == int
	assert numpy.ndim(U) == 2
	U = numpy.asarray(U,dtype=float)

	return _adolc.hov_reverse(tape_tag, D, U)


def hov_ti_reverse(tape_tag, U):
	"""
	higher order vector reverse:
	(Z,nz) = hov_ti_reverse(tape_tag, U)
	F:R^N -> R^M
	D is the highest order of the derivative
	U is (Q x M x D+1)-matrix, Q adjoint directions
	Z is (Q x N x D+1)-matrix, adjoint directional derivative Z = [U F'(x), U F\" v[:,0],  U F\" v[:,1] + 0.5 U F^(3) v[:,0],... ]
	nz is (Q x N)-matrix, information about the sparsity of Z:
	0:trivial, 1:linear, 2:polynomial, 3:rational, 4:transcendental, 5:non-smooth
	after calling fos_forward or hos_forward with keep = D+1
	"""
	assert type(tape_tag) == int
	assert numpy.ndim(U) == 3
	(Q,M,Dp1) = numpy.shape(U)
	ts = tapestats(tape_tag)
	
	assert ts['NUM_DEPENDENTS'] == M
	
	U = numpy.asarray(U,dtype=float)
	return _adolc.hov_ti_reverse(tape_tag, U)
	

def tape_to_latex(tape_tag,x,y):
	"""
,	"\n\ntape_to_latex(tape_tag,x,y)
											"F:R^N -> R^M
											"x is N-vector  y is M-vector\n
											"writes the tape to a file called tape_x.tex that can be compile with Latex\n
											

	"""
	assert type(tape_tag) == int
	x = numpy.asarray(x, dtype=float)
	y = numpy.asarray(y, dtype=float)

	return _adolc.tape_to_latex(tape_tag, x, y)

def tapestats(tape_tag):
	"""
	returns a dictionary with information on the tape:
	NUM_INDEPENDENTS,                          /* # of independent variables */ 
	NUM_DEPENDENTS,                              /* # of dependent variables */ 
	NUM_MAX_LIVES,                                /* max # of live variables */ 
	TAY_STACK_SIZE,               /* # of values in the taylor (value) stack */ 
	OP_BUFFER_SIZE,   /* # of operations per buffer == OBUFSIZE (usrparms.h) */ 
	NUM_OPERATIONS,                               /* overall # of operations */ 
	OP_FILE_ACCESS,                        /* operations file written or not */ 
	NUM_LOCATIONS,                                 /* overall # of locations */ 
	LOC_FILE_ACCESS,                        /* locations file written or not */ 
	NUM_VALUES,                                       /* overall # of values */ 
	VAL_FILE_ACCESS,                           /* values file written or not */ 
	LOC_BUFFER_SIZE,   /* # of locations per buffer == LBUFSIZE (usrparms.h) */ 
	VAL_BUFFER_SIZE,      /* # of values per buffer == CBUFSIZE (usrparms.h) */ 
	TAY_BUFFER_SIZE,     /* # of taylors per buffer <= TBUFSIZE (usrparms.h) */ 
	"""
	assert type(tape_tag) == int
	return _adolc.tapestats(tape_tag)