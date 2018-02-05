import numpy
import math
import adolc

# tape a function evaluation
ax = numpy.array([adolc.adouble(0.) for n in range(2)])
# ay = adolc.adouble(0)
adolc.trace_on(13)
adolc.independent(ax)
ay = numpy.sin(ax[0] + ax[1]*ax[0])
adolc.dependent(ay)
adolc.trace_off()

x = numpy.array([3., 7.])
y = numpy.zeros(1)
adolc.tape_to_latex(13, x, y)

y = adolc.function(13, x)
g = adolc.gradient(13, x)
J = adolc.jacobian(13, x)

print('function y=', y)
print('gradient g=', g)
print('Jacobian J=', J)


