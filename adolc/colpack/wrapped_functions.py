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
    
    if options is None:
        options = numpy.array([1,1,0,0], dtype=numpy.int32)

    options = numpy.asarray(options,dtype=numpy.int32)
    assert numpy.ndim(options) == 1
    assert numpy.size(options) == 4

    assert numpy.ndim(x) == 1
    x = numpy.asarray(x, dtype=float)

    return _colpack.sparse_jac_no_repeat(tape_tag, x, options)

def sparse_jac_repeat(tape_tag, x, nnz, rind, cind, values):
    """
    computes sparse Jacobian J for a function F:R^N -> R^M with
    the sparsity pattern that has been computed previously (e.g. by calling sparse_jac_no_repeat)

    I guess it also reuses the options that have been set previously. So it would be not necessary to set the options again.

    
    [nnz, rind, cind, values] = sparse_jac_repeat(tape_tag, x, rind, cind, values)

    INPUT:
    The base point x at which the Jacobian should be computed, i.e. J = dF(x)/dx
    
    OUTPUT:
    nnz is the guessed number of nonzeros in the Jacobian. This can be larger than the true number of nonzeros.

    sparse matrix representation in standard format:
    rind is an nnz-array of row indices
    cind is an nnz-array of column indices
    values are the corresponding Jacobian entries
    """
    assert type(tape_tag) == int
    assert type(nnz) == int

    assert numpy.ndim(x) == 1
    assert numpy.ndim(rind) == 1
    assert numpy.ndim(cind) == 1
    assert numpy.ndim(values) == 1

    x = numpy.asarray(x, dtype=float)
    rind= numpy.asarray(rind, dtype=numpy.uint32)
    cind= numpy.asarray(cind, dtype=numpy.uint32)
    values = numpy.asarray(values, dtype=float)

    return _colpack.sparse_jac_repeat(tape_tag, x, nnz, rind, cind, values)


def sparse_hess_no_repeat(tape_tag, x, options = None):
    """
    computes sparse Hessian for a function F:R^N -> R without any prior information,

    [nnz, rind, cind, values] =sparse_hess_no_repeat(tape_tag, x, options)

    INPUT:
    The base point x at which the Jacobian should be computed, i.e. J = dF(x)/dx
    options is a list or array of length 2
                    options[0] :test the computational graph control flow
                            0 - safe mode (default)
                            1 - tight mode
                    options[1] : way of recovery
                            0 - indirect recovery
                            1 - direct recovery

    OUTPUT:
    nnz is the guessed number of nonzeros in the Hessian. This can be larger than the true number of nonzeros.

    sparse matrix representation in standard format:
    rind is an nnz-array of row indices
    cind is an nnz-array of column indices
    values are the corresponding Jacobian entries
    """
    assert type(tape_tag) == int
    
    if options is None:
        options = numpy.array([0,0], dtype=numpy.int32)

    options = numpy.asarray(options,dtype=numpy.int32)
    assert numpy.ndim(options) == 1
    assert numpy.size(options) == 2

    assert numpy.ndim(x) == 1
    x = numpy.asarray(x, dtype=float)
    return _colpack.sparse_hess_no_repeat(tape_tag, x, options)

def sparse_hess_repeat(tape_tag, x, rind, cind, values):
    """
    computes sparse Hessian for a function F:R^N -> R without any prior information,

    [nnz, rind, cind, values] =sparse_hess_no_repeat(tape_tag, x, options)

    INPUT:
    The base point x at which the Jacobian should be computed, i.e. J = dF(x)/dx
    options is a list or array of length 2
                    options[0] :test the computational graph control flow
                            0 - safe mode (default)
                            1 - tight mode
                    options[1] : way of recovery
                            0 - indirect recovery
                            1 - direct recovery

    OUTPUT:
    nnz is the guessed number of nonzeros in the Hessian. This can be larger than the true number of nonzeros.

    sparse matrix representation in standard format:
    rind is an nnz-array of row indices
    cind is an nnz-array of column indices
    values are the corresponding Jacobian entries
    """
    assert type(tape_tag) == int

    assert numpy.ndim(x)      == 1
    assert numpy.ndim(rind)   == 1
    assert numpy.ndim(cind)   == 1
    assert numpy.ndim(values) == 1

    nnz = int(numpy.size(rind))

    assert nnz == numpy.size(cind)
    assert nnz == numpy.size(values)

    x      = numpy.asarray(x, dtype=float)
    rind   = numpy.asarray(rind, dtype=numpy.uint32)
    cind   = numpy.asarray(cind, dtype=numpy.uint32)
    values = numpy.asarray(values, dtype=float)

    return _colpack.sparse_hess_repeat(tape_tag, x, nnz, rind, cind, values)