import numpy

class adouble:
    def __init__(self,x):
       self.x = x
    
    
    def __mul__(self, rhs):
       print('called __mul__')
       if isinstance(rhs, adouble):
           print('case adouble')
           return adouble(self.x * rhs.x)
       elif numpy.isscalar(rhs):
           print('case scalar')
           return adouble(self.x * rhs)
       elif isinstance(rhs, numpy.ndarray):
           print('case ndarray')
           return rhs * self

    def __rmul__(self, lhs):
       print('called __rmul__')
       return self * lhs
    
    def __str__(self):
       return str(self.x)+'a'
    
    def __repr__(self):
       return str(self)

print('numpy.__version__=',numpy.__version__,'\n')

x  = numpy.ones(2,dtype=float)
z = numpy.ones(2,dtype=float)
y = numpy.ones(2,dtype=object)


a = adouble(2)

print('executing a *= x')
a *= x
print(a,' where expected [2.0a, 2.0a]\n')

print('executing y *= a')
y *= a
print(y,' where expected [2.0a, 2.0a]\n')

print('executing z = z * a')
z = z * a
print(z,' where expected [2.0a, 2.0a]\n')

print('executing x *= a')
x *= a
print(x,' where expected [2.0a, 2.0a]\n')


