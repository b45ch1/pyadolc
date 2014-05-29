import numpy
from . import _adolc
from ._adolc import *

def adouble(x):
    """
    Return adouble from scalar or array of arbitrary shape
    INPUT:  either a float or an array of floats
    OUTPUT: adouble or array of adoubles
    """
    if numpy.isscalar(x):
        return _adolc.adouble(float(x))
    elif isinstance(x,_adolc.adouble) or isinstance(x,_adolc.adub):
        return _adolc.adouble(x)

    else:
        x = numpy.ascontiguousarray(x, dtype=float)
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
        list(map(_adolc.adouble.is_independent,axr,xr))
        return ax


def dependent(ax):
    """
    dependent(ax)
    INPUT :  numpy.array of type adouble
    OUTPUT:  ax

    Mark ax as dependent.
    """
    if isinstance(ax, _adolc.adouble) or isinstance(ax, _adolc.adub):
        depends_on(ax)
        return ax
    else:
        axr = numpy.ravel(ax)
        N   = numpy.size(axr)

        for n in range(N):
            if numpy.isscalar(axr[n]):
                depends_on(adouble(axr[n]))
            else:
                depends_on(axr[n])
        return ax


def trace_on(tape_tag):
    """
    start recording to the tape with index tape_tag
    """
    assert type(tape_tag) == int
    return _adolc.trace_on(tape_tag)

def trace_off():
    """turn off tracing"""
    return _adolc.trace_off()



def function(tape_tag,x):
    """
    evaluate the function f(x) recorded on tape with index tape_tag
    """
    assert type(tape_tag) == int
    ts = tapestats(tape_tag)
    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']
    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1
    y = numpy.zeros(M, dtype=float)
    _adolc.function(tape_tag, M, N, x, y)
    return y

def gradient(tape_tag,x):
    """
    evaluate the gradient g = f'(x), f:R^N -> R
    """
    assert type(tape_tag) == int
    ts = tapestats(tape_tag)
    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']
    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1
    g = numpy.zeros(N, dtype=float)
    _adolc.gradient(tape_tag, N, x, g)
    return g

def hessian(tape_tag, x, format='full' ):
    """
    evaluate the hessian H = f\"(x), f:R^N -> R"
    """
    assert type(tape_tag) == int
    ts = tapestats(tape_tag)
    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']
    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1

    if format == 'full':
        H = numpy.zeros((N,N), dtype=float)
        ints = [i for i in range(N)]
        _adolc.hessian(tape_tag, N, x, H)
        H[:] = H[:] + H.T
        H[ints, ints] /= 2.
        return H
    else:
        raise NotImplementedError('hessian can only return full matrices ATM')

def jacobian(tape_tag,x):
    """
    evaluate the jacobian J = F'(x), F:R^N -> R^M
    """
    assert type(tape_tag) == int
    ts = tapestats(tape_tag)
    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']
    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1
    J = numpy.zeros((M,N), dtype=float)
    _adolc.jacobian(tape_tag, M, N, x, J)
    return J


def vec_jac(tape_tag, x, u, repeat = False):
    """
    evaluate u^T F'(x), F:R^N -> R^M
    """
    assert type(repeat) == bool or type(repeat) == int
    assert type(tape_tag) == int
    ts = tapestats(tape_tag)
    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']
    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1
    u = numpy.ascontiguousarray(u, dtype=float)
    assert numpy.size(u) == M
    assert numpy.ndim(u) == 1
    z = numpy.zeros(N, dtype=float)
    _adolc.vec_jac(tape_tag, M, N, repeat, x, u, z)
    return z

def jac_vec(tape_tag, x, v):
    """
    evaluate  F'(x)v, F:R^N -> R^M
    """
    ts = tapestats(tape_tag)
    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']
    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1
    v = numpy.ascontiguousarray(v, dtype=float)
    assert numpy.size(v) == N
    assert numpy.ndim(v) == 1
    z = numpy.zeros(M, dtype=float)
    _adolc.jac_vec(tape_tag, M, N, x, v, z)
    return z

def hess_vec(tape_tag, x, v):
    """
    evaluate  f''(x)v, f:R^N -> R
    """
    ts = tapestats(tape_tag)
    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']
    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1
    v = numpy.ascontiguousarray(v, dtype=float)
    assert numpy.size(v) == N
    assert numpy.ndim(v) == 1
    z = numpy.zeros(N, dtype=float)
    _adolc.hess_vec(tape_tag, N, x, v, z)
    return z


def lagra_hess_vec(tape_tag, x, u, v):
    """
    evaluate z = u^T F''(x)v, F:R^N -> R^M

    v N-array
    u M-array
    z N-array
    """
    ts = tapestats(tape_tag)
    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1

    v = numpy.ascontiguousarray(v, dtype=float)
    assert numpy.size(v) == N
    assert numpy.ndim(v) == 1

    u = numpy.ascontiguousarray(u, dtype=float)
    assert numpy.size(u) == M
    assert numpy.ndim(u) == 1

    z = numpy.zeros(N, dtype=float)

    _adolc.lagra_hess_vec(tape_tag, M, N, x, v, u, z)
    return z

def zos_forward(tape_tag,x,keep):
    """
    zero order scalar forward:
    y = zos_forward(tape_tag,x,keep)
    F:R^N -> R^M
    x is N-vector, y is M-vector
    keep = 1 prepares for fos_reverse or fov_reverse
    """
    assert type(keep) == int
    ts = tapestats(tape_tag)
    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1

    y = numpy.zeros(M, dtype=float)
    _adolc.zos_forward(tape_tag, M, N, keep, x, y)

    return y

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
    ts = tapestats(tape_tag)

    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1

    v = numpy.ascontiguousarray(v, dtype=float)
    assert numpy.size(v) == N
    assert numpy.ndim(v) == 1

    y = numpy.zeros(M, dtype=float)
    w = numpy.zeros(M, dtype=float)
    _adolc.fos_forward(tape_tag, M, N, keep, x, v, y, w)

    return (y,w)

def fov_forward(tape_tag, x, V):
    """
    first order vector forward:
    (y,W) = fov_forward(tape_tag, x, V)
    F:R^N -> R^M
    x is N-vector, y is M-vector
    V is (N x P)-matrix. P directions
    W is (M x P)-matrix. P directiona derivatives
    """

    assert type(tape_tag) == int
    ts = tapestats(tape_tag)

    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1

    V = numpy.ascontiguousarray(V, dtype=float)
    assert numpy.shape(V)[0] == N
    assert numpy.ndim(V) == 2
    P = numpy.shape(V)[1]

    y = numpy.zeros(M, dtype=float)
    W = numpy.zeros((M,P), dtype=float)

    _adolc.fov_forward(tape_tag, M, N, P, x, V, y, W)

    return (y,W)

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
    assert type(keep) == int
    ts = tapestats(tape_tag)

    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1

    V = numpy.ascontiguousarray(V, dtype=float)
    assert numpy.shape(V)[0] == N
    assert numpy.ndim(V) == 2
    D = numpy.shape(V)[1]

    y = numpy.zeros(M, dtype=float)
    W = numpy.zeros((M,D), dtype=float)

    _adolc.hos_forward(tape_tag, M, N, D, keep, x, V, y, W)

    return (y,W)

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
    ts = tapestats(tape_tag)

    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1

    V = numpy.ascontiguousarray(V, dtype=float)
    assert numpy.shape(V)[0] == N
    assert numpy.ndim(V) == 3
    P = numpy.shape(V)[1]
    D = numpy.shape(V)[2]


    y = numpy.zeros(M, dtype=float)
    W = numpy.zeros((M,P,D), dtype=float)

    _adolc.hov_forward(tape_tag, M, N, D, P, x, V, y, W)
    return (y,W)

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
    assert type(keep) == int
    assert type(tape_tag) == int
    ts = tapestats(tape_tag)

    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    x = numpy.ascontiguousarray(x, dtype=float)
    assert numpy.size(x) == N
    assert numpy.ndim(x) == 1

    V = numpy.ascontiguousarray(V, dtype=float)
    assert numpy.shape(V)[0] == N
    assert numpy.ndim(V) == 3
    P = numpy.shape(V)[1]
    D = numpy.shape(V)[2]


    y = numpy.zeros(M, dtype=float)
    W = numpy.zeros((M,P,D), dtype=float)

    _adolc.hov_wk_forward(tape_tag, M, N, D, keep, P, x, V, y, W)
    return (y,W)
    # raise NotImplementedError


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
    ts = tapestats(tape_tag)

    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    u = numpy.ascontiguousarray(u, dtype=float)
    assert numpy.size(u) == M
    assert numpy.ndim(u) == 1

    z = numpy.zeros(N, dtype=float)
    _adolc.fos_reverse(tape_tag, M, N, u, z)
    return z


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
    ts = tapestats(tape_tag)

    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    U = numpy.ascontiguousarray(U, dtype=float)
    assert numpy.ndim(U) == 2
    assert numpy.shape(U)[1] == M
    Q = numpy.shape(U)[0]

    Z = numpy.zeros((Q,N), dtype=float)
    _adolc.fov_reverse(tape_tag, M, N, Q, U, Z)
    return Z

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

    ts = tapestats(tape_tag)

    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    u = numpy.ascontiguousarray(u, dtype=float)
    assert numpy.size(u) == M
    assert numpy.ndim(u) == 1
    Z = numpy.zeros((N, D+1), dtype=float)
    _adolc.hos_reverse(tape_tag, M, N, D, u, Z)
    return Z

def hos_ti_reverse(tape_tag, U):
    """
    higher order scalar reverse:
    Z = hos_ti_reverse(tape_tag, U)
    F:R^N -> R^M
    U is (M x D+1)-matrix,
    Z is (N x D+1)-matrix, adjoint directional derivative Z = [U F'(x), U F\" v[:,0],  U F\" v[:,1] + 0.5 U F^(3) v[:,0],... ]
    D is the highest order of the derivative
    after calling fos_forward or hos_forward with keep = D+1
    """
    assert type(tape_tag) == int

    ts = tapestats(tape_tag)

    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    U = numpy.ascontiguousarray(U, dtype=float)
    assert numpy.ndim(U) == 2
    assert numpy.shape(U)[0] == M
    Dp1 = numpy.shape(U)[1]
    D = Dp1 - 1
    Z = numpy.zeros((N, D+1), dtype=float)
    _adolc.hos_ti_reverse(tape_tag, M, N, D, U, Z)
    return Z


def hov_reverse(tape_tag, D, U):
    """
    this function is deprecated, use hov_ti_reverse instead!

    higher order vector reverse:
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

    ts = tapestats(tape_tag)
    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    U = numpy.ascontiguousarray(U, dtype=float)
    assert numpy.ndim(U) == 2
    assert numpy.shape(U)[1] == M
    Q = numpy.shape(U)[0]

    Z = numpy.zeros((Q, N, D+1), dtype=float)
    nz = numpy.zeros((Q,N), dtype=numpy.int16)

    _adolc.hov_reverse(tape_tag, M, N, D, Q, U, Z, nz)

    return (Z, nz)


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

    ts = tapestats(tape_tag)
    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    U = numpy.ascontiguousarray(U, dtype=float)
    assert numpy.ndim(U) == 3
    assert numpy.shape(U)[1] == M
    Q = numpy.shape(U)[0]
    Dp1 = numpy.shape(U)[2]
    D = Dp1 - 1

    Z = numpy.zeros((Q, N, Dp1), dtype=float)
    nz = numpy.zeros((Q,N), dtype=numpy.int16)
    _adolc.hov_ti_reverse(tape_tag, M, N, D, Q, U, Z, nz)
    return (Z,nz)

def hos_ov_reverse(tape_tag, P, U):
    """
    higher order scalar reverse on vector keep.
    Z = hos_ov_reverse(tape_tag, U)
    F:R^N -> R^M
    U is (M x D+1)-matrix,
    Z is (N x P x D+1)-matrix, adjoint directional derivative Z = [U F'(x), U F\" v[:,0],  U F\" v[:,1] + 0.5 U F^(3) v[:,0],... ]
    D is the highest order of the derivative
    P is the number of directions saved in the forward run
    after calling hov_wk_forward with keep = D+1
    """
    assert type(tape_tag) == int
    assert type(P) == int

    ts = tapestats(tape_tag)

    N = ts['NUM_INDEPENDENTS']
    M = ts['NUM_DEPENDENTS']

    U = numpy.ascontiguousarray(U, dtype=float)
    assert numpy.ndim(U) == 2
    assert numpy.shape(U)[0] == M
    Dp1 = numpy.shape(U)[1]
    D = Dp1 - 1

    Z = numpy.zeros((N, P, D+1), dtype=float)
    _adolc.hos_ov_reverse(tape_tag, M, N, D, P, U, Z)
    return Z

def condassign(x, cond, y, z = None):
    """

    x = condassign(cond, y, z = None)

    equivalent to:
    if cond:
       x = y

    else:
        if z != None:
            x = z

    """

    def c(v):
        if isinstance(v, _adolc.adub):
            return _adolc.adouble(v)
        return v

    x = c(x)
    cond = c(cond)
    y = c(y)

    if z == None:
        return _adolc.condassign(x, cond, y)
    else:
        z = c(z)
        return _adolc.condassign(x, cond, y, z)


def tape_to_latex(tape_tag,x,y):
    """
    tape_to_latex(tape_tag,x,y)
    F:R^N -> R^M
    x is N-vector  y is M-vector
    writes the tape to a file called tape_x.tex that can be compile with Latex
    """
    assert type(tape_tag) == int
    x = numpy.ascontiguousarray(x, dtype=float)
    y = numpy.ascontiguousarray(y, dtype=float)

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