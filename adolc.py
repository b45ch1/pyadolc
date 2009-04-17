# -*- coding: utf-8 -*-
import numpy
import _adolc
from _adolc import *

__doc__ = _adolc.__doc__

def adouble(x):
	"""
	Return adouble from scalar or array of arbitrary shape
	INPUT:  either a float or an array of floats
	OUTPUT: adouble or array of adoubles
	"""
	if numpy.isscalar(x) or isinstance(x,_adolc.adouble):
		return _adolc.adouble(x)
	else:
		x = numpy.asarray(x)
		shp = numpy.shape(x)
		xr = numpy.ravel(x)
		axr = numpy.array([_adolc.adouble(xr[n]) for n in range(len(xr))])
		ax = axr.reshape(shp)
		return ax


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
			depends_on(axr[n])

		return ax
		