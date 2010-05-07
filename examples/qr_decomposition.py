from adolc import *
import adolc
import adolc.linalg
from numpy import *

N = 4
A = array([[ r*N + c for c in range(N) ] for r in range(N)],dtype=float)

aA = adouble(A)
trace_on(1)
independent(aA)
(aQT, aR) = adolc.linalg.qr(aA)
dependent(aQT)
dependent(aR)
trace_off()

V = zeros((N**2,1,1),dtype=float)
V[:,0,0] = 1.
(result,Z) = hov_forward(1, ravel(A), V)
QT = result[:N**2].reshape((N,N))
R  = result[N**2:].reshape((N,N))

#(QT,R) = myqr(A)
#print dot(QT.T,R) - A

QTdot = Z[:N**2,0,0].reshape((N,N))
Rdot  = Z[N**2:,0,0].reshape((N,N))

print QTdot.T + dot(dot(QT.T,QTdot), QT.T)
