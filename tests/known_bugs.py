import numpy
import adolc
import os


class oofun:
   __array_priority__ = 1000
   def __init__(self,x):
       self.x = x

   def __mul__(self, rhs):
       print 'called __mul__'
       if isinstance(rhs, oofun):
           return oofun(self.x * rhs.x)
       else:
           return rhs * self

   def __rmul__(self, lhs):
       print 'called __rmul__'
       return oofun(self.x * lhs)

   def __str__(self):
       return str(self.x)+'a'

   def __repr__(self):
       return str(self)

def test_numpy_array_imul_operations():
    
    x = numpy.ones(1)
    y = 1.
    b = oofun(2)
    x *= b
    y *= b
    
    
    
    print x,y





if __name__ == '__main__':
    try:
        import nose
    except:
        print 'Please install nose for unit testing'
    nose.runmodule()
