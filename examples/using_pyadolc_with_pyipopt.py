"""
This test checks
1. the sparse functionality of pyadolc
2. the execution speed compared to the direct sparse computation
3. run the optimization with the derivatives provided by pyadolc

IPOPT is an interior point algorithm to solve

min     f(x)
    x in R^n
s.t.       g_L <= g(x) <= g_U
            x_L <=  x   <= x_U

this test fails probably because of a bug in pyipopt 

"""



import numpy, scipy, scipy.sparse
import adolc, pyipopt, time
from numpy.testing import *


def eval_f(x, user_data = None):
    """ objective function """
    assert len(x) == 4
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

def eval_grad_f(x, user_data = None):
    """ symbolic gradient of  the objective function """
    assert len(x) == 4
    grad_f = numpy.array([
        x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]) ,
        x[0] * x[3],
        x[0] * x[3] + 1.0,
        x[0] * (x[0] + x[1] + x[2])
        ])
    return grad_f;
    
def eval_g(x, user_data= None):
    """ constraint function """
    assert len(x) == 4
    return numpy.array([
        x[0] * x[1] * x[2] * x[3], 
        x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]
    ])
    
def eval_lagrangian(x, lagrange, obj_factor, user_data = None):
    return obj_factor * eval_f(x) + numpy.dot(lagrange, eval_g(x))

nnzj = 8
def eval_jac_g(x, flag, user_data = None):
    """ sparse jacobian of constraint function and sparsity pattern"""
    
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
    """ sparse hessian of the lagrangian and sparsity pattern"""
    
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

# trace objective function
adolc.trace_on(1)
ax = adolc.adouble(x0)
adolc.independent(ax)
ay = eval_f(ax)
adolc.dependent(ay)
adolc.trace_off()

# trace constraint function
adolc.trace_on(2)
ax = adolc.adouble(x0)
adolc.independent(ax)
ay = eval_g(ax)
adolc.dependent(ay)
adolc.trace_off()

# trace lagrangian function
adolc.trace_on(3)
ax = adolc.adouble(x0)
alagrange = adolc.adouble([1.,1.])
aobj_factor = adolc.adouble(1.)
adolc.independent(ax)
adolc.independent(alagrange)
adolc.independent(aobj_factor)
ay = eval_lagrangian(ax, alagrange, aobj_factor)
adolc.dependent(ay)
adolc.trace_off()

def eval_f_adolc(x, user_data = None):
    return adolc.function(1,x)[0]

def eval_grad_f_adolc(x, user_data = None):
    return adolc.gradient(1,x)

def eval_g_adolc(x, user_data= None):
    return adolc.function(2,x)

class Eval_jac_g_adolc:
    
    def __init__(self, x):
        options = numpy.array([1,1,0,0],dtype=int)
        result = adolc.colpack.sparse_jac_no_repeat(2,x,options)
        
        self.nnz  = result[0]     
        self.rind = numpy.asarray(result[1],dtype=int)
        self.cind = numpy.asarray(result[2],dtype=int)
        self.values = numpy.asarray(result[3],dtype=float)
        
    def __call__(self, x, flag, user_data = None):
        if flag:
            return (self.rind, self.cind)
        else:
            result = adolc.colpack.sparse_jac_repeat(2, x, self.nnz, self.rind,
                self.cind, self.values)
            return result[3]


class Eval_h_adolc:
    
    def __init__(self, x):
        options = numpy.array([0,0],dtype=int)
        result = adolc.colpack.sparse_hess_no_repeat(3,x,options)
        
        self.rind = numpy.asarray(result[1],dtype=int)
        self.cind = numpy.asarray(result[2],dtype=int)
        self.values = numpy.asarray(result[3],dtype=float)
        
        # need only upper left part of the Hessian
        self.mask = numpy.where(self.cind < 4)
        
    def __call__(self, x, lagrange, obj_factor, flag, user_data = None):
        if flag:
            return (self.rind[self.mask], self.cind[self.mask])
        else:
            x = numpy.hstack([x,lagrange,obj_factor])
            result = adolc.colpack.sparse_hess_repeat(3, x, self.rind,
                self.cind, self.values)
            return result[3][self.mask]


# create callable instance of the classes
eval_jac_g_adolc = Eval_jac_g_adolc(x0)
eval_h_adolc = Eval_h_adolc(x0)

pat = eval_h(x0,  numpy.array([1.,2.]), 1., True)
val = eval_h(x0,  numpy.array([1.,2.]), 1., False)
H1 = scipy.sparse.coo_matrix( (val, (pat[0], pat[1])), shape=(4, 4))
print 'symbolic Hessian=\n',H1

pat = eval_h_adolc(x0,  numpy.array([1.,2.]), 1., True)
val = eval_h_adolc(x0,  numpy.array([1.,2.]), 1., False)
H2 = scipy.sparse.coo_matrix( (val, (pat[0], pat[1])), shape=(4, 4))
print 'pyadolc Hessian=\n',H2

# function of f
assert_almost_equal(eval_f(x0), eval_f_adolc(x0))

# gradient of f
assert_array_almost_equal(eval_grad_f(x0), eval_grad_f_adolc(x0))

# function of g
assert_array_almost_equal(eval_g(x0), adolc.function(2,x0))

# sparse jacobian of g
assert_array_equal(eval_jac_g_adolc(x0,True)[0], eval_jac_g(x0,True)[0])
assert_array_equal(eval_jac_g_adolc(x0,True)[1], eval_jac_g(x0,True)[1])
assert_array_equal(eval_jac_g_adolc(x0,False),  eval_jac_g(x0,False))


# test optimization with PYIPOPT
nvar = 4
x_L = numpy.ones((nvar), dtype=numpy.float_) * 1.0
x_U = numpy.ones((nvar), dtype=numpy.float_) * 5.0

ncon = 2
g_L = numpy.array([25.0, 40.0])
g_U = numpy.array([2.0*pow(10.0, 19), 40.0]) 

nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h)
start_time = time.time()
result =  nlp.solve(x0)
end_time = time.time()
nlp.close()
pure_python_optimization_time = end_time - start_time


nlp_adolc = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f_adolc, eval_grad_f_adolc, eval_g_adolc, eval_jac_g_adolc, eval_h_adolc)

start_time = time.time()
result_adolc = nlp_adolc.solve(x0)
end_time = time.time()
nlp_adolc.close()

adolc_optimization_time = end_time - start_time
print 'optimization time with derivatives computed by adolc = ', adolc_optimization_time
print 'optimization time with derivatives computed by hand = ',pure_python_optimization_time

# # this works with the pyipopt version from code.google.com
# assert_array_almost_equal(result[0], result_adolc[0])
# assert_array_almost_equal(result[1], result_adolc[1])
# assert_array_almost_equal(result[2], result_adolc[2])
# assert_array_almost_equal(result[3], result_adolc[3])

#this works with the pyipopt version from github by alanfalloon
assert_array_almost_equal(result['x'],result_adolc['x'])
assert_array_almost_equal(result['mult_xL'],result_adolc['mult_xL'])
assert_array_almost_equal(result['mult_xU'],result_adolc['mult_xU'])
assert_array_almost_equal(result['mult_g'],result_adolc['mult_g'])
assert_array_almost_equal(result['f'],result_adolc['f'])

