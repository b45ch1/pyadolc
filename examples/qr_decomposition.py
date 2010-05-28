from adolc import *
import adolc
import adolc.linalg
from numpy import *

N = 3

A    = numpy.array([[1,2,3],[0,4,5],[0,0,6]],dtype=float)
Adot = numpy.array([[3,7,5],[1,7,2],[7,3,5]],dtype=float)
# print numpy.linalg.svd(A)[1]

aA = adouble(A)
trace_on(1)
independent(aA)
(aQ, aR) = adolc.linalg.qr(aA)
dependent(aQ)
dependent(aR)
trace_off()

V = zeros((N**2,1,1),dtype=float)
V[:,0,0] = Adot.ravel()

(result,Z) = hov_forward(1, ravel(A), V)
Q = result[:N**2].reshape((N,N))
R  = result[N**2:].reshape((N,N))

Adot = V.reshape((N,N))
Qdot = Z[:N**2,0,0].reshape((N,N))
Rdot  = Z[N**2:,0,0].reshape((N,N))

print dot(Qdot, R) + dot(Q, Rdot)

print 'd=0: QR - A :\n', dot(Q, R) - A
print 'd=1: QR - A :\n', dot(Qdot, R) + dot(Q, Rdot) - Adot

print 'd=0: Q.T Q - I :\n', dot(Q.T, Q) - numpy.eye(N)
print 'd=1: Q.T Q - I :\n', dot(Qdot.T, Q) + dot(Q.T,Qdot)

print 'd=0: R - triu(R) :\n', R - numpy.triu(R)
print 'd=1: R - triu(R) :\n', Rdot - numpy.triu(Rdot)



