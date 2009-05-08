import _colpack
from _colpack import *

import numpy



def sparse_jac_no_repeat(tape_tag, x, options):
	"""
	computes sparse Jacobian for a function F:R^N -> R^M without any prior information,
	i.e. this function internally finds the sparsity pattern
	of the Jacobian J, then uses Graph Coloring to find the smallest number P or Q of necessary directions
	(P in the forward mode, Q in the reverse mode)
	and computes dot(J,S) with S (N,P) array in forward mode
	or dot(S^T,J) with S(N,Q) in the reverse mode
	
	[nnz, rind, cind, values] =sparse_jac_no_repeat(tape_tag, x, options)

	INPUT:
	The base point x at which the Jacobian should be computed, i.e. J = dF(x)/dx
	options is a list or array of length 4
		options[0] : way of sparsity pattern computation
				0 - propagation of index domains (default)
				1 - propagation of bit pattern
		options[1] : test the computational graph control flow
				0 - safe mode (default)
				1 - tight mode
		options[2] : way of bit pattern propagation
				0 - automatic detection (default)
				1 - forward mode
				2 - reverse mode
		options[3] : way of compression
				0 - column compression (default)
				1 - row compression

	OUTPUT:
	nnz is the guessed number of nonzeros in the Jacobian. This can be larger than the true number of nonzeros.

	sparse matrix representation in standard format:
	rind is an nnz-array of row indices
	cind is an nnz-array of column indices
	values are the corresponding Jacobian entries
	"""
	assert type(tape_tag) == int

	options = numpy.asarray(options,dtype=int)
	assert numpy.ndim(options) == 1
	assert numpy.size(options) == 4

	assert numpy.ndim(x) == 1
	x = numpy.asarray(x, dtype=float)

	return _colpack.sparse_jac_no_repeat(tape_tag, x, options)

	