import numpy
import _adolc
from _adolc import *

__doc__ = _adolc.__doc__

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
	if numpy.isscalar(ax):
		x = ax.val
		ax.is_independent(x)
		return ax
	else:
		axr = numpy.ravel(ax)
		N   = numpy.size(axr)
		xr = numpy.array([axr[n].val for n in range(N)])
		map(adouble.is_independent,axr,xr)
		return ax
	

def dependent(ax):
	"""
	dependent(ax)
	INPUT :  numpy.array of type adouble
	OUTPUT:  ax
	
	Mark ax as dependent.
	"""
	if numpy.isscalar(ax):
		depends_on(ax)
		return ax
	else:
		axr = numpy.ravel(ax)
		N   = numpy.size(axr)

		for n in range(N):
			depends_on(axr[n])

		return ax
		