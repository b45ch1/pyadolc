import numpy
import adolc._adolc

class Tangent:
    """
    Simple class to emulate multivariate Taylor propagation. E.g. this class makes it possible to:
    
        * compute a Hessian matrix H without using exact interpolation.
        * using UTPS of functions that are derivatives of other functions, e.g. 
         if f:R^Nx x R^Np --> R^M  then let J(x) = df/dp(x,p) be the Jacobian of f(x,p). Now we want to use UTPS in x using ADOL-C.
    
    Usage example::
        
        from adolc import *
        
        def f(x,p):
            return x**2 *(p[0] + p[1])
    
        AP = AdolcProgram()
        AP.trace_on(1)
        ax = adouble(3.)
        ap = adouble([5.,7.])
        AP.independent(ax)
        AP.independent(ay)
        
        tp = Tangent(ap,1)
        tf = f(ax,tp)
        
        aJ = tf.xdot
        
        AP.dependent(aJ)
        AP.trace_off()
    """
    
    def __init__(self, x, xdot):
        
        self.x = x
        self.xdot = xdot
        
    def clone(self):
        if numpy.isscalar(self.x):
            x = self.x
        
        else:
            x = self.x.clone()
        
        if numpy.isscalar(self.xdot):
            xdot = self.xdot
        
        else:
            xdot = self.xdot.clone()
            
        return Tangent(x,xdot)
        
    def __add__(self, rhs):
        retval = self.clone()
        retval += rhs
        return retval
    


    def __mul__(self, rhs):
        retval = self.clone()
        retval *= rhs
        return retval        

    def __iadd__(self,rhs):
        if isinstance(rhs, Tangent):
           self.x = self.x + rhs.x
           self.xdot += rhs.xdot
        
        else:
            self.x += rhs
            
        return self
        
    def __imul__(self,rhs):
        if isinstance(rhs, Tangent):
            self.xdot *= rhs.x
            self.xdot += self.x * rhs.xdot
            self.x *= rhs.x
        
        else:
            self.xdot *= rhs
            self.x *= rhs
            
        return self        



    def __radd__(self, lhs):
        return self + lhs
        
    def __rmul__(self, lhs):
        return self * lhs

    def __str__(self):
        return "[%s,%s]"%(str(self.x),str(self.xdot))
        
    def __repr__(self):
        return str(self)
