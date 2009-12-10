"""
This module provides linear algebra routines that can be differentiated with pyadolc.

Warning:

    these routines are just for testing purposes. The algorithms are not:

        * tuned for efficiency
        * tuned to be stable (e.g. no pivoting...)

"""
import adolc
import numpy

def qr(in_A):
    """
    QR decomposition of A
   
    Q,R = qr(A)
    
    """
    # input checks
    Ndim = numpy.ndim(in_A)
    assert Ndim == 2
    N,M = numpy.shape(in_A)
    assert N==M
    
    assert isinstance(in_A[0,0], adolc._adolc.adouble)

    # prepare R and QT
    R  = in_A.copy()
    QT = numpy.array([[adolc.adouble(0) for c in range(N)] for r in range(N) ])
    
    for n in range(N):
        QT[n,n]  += 1

    # main algorithm
    for n in range(N):
        for m in range(n+1,N):
            a = R[n,n]
            b = R[m,n]
            r = numpy.sqrt(a**2 + b**2)
            c = a/r
            s = b/r

            for k in range(N):
                Rnk = R[n,k]
    
                R[n,k] = c*Rnk + s*R[m,k]
                R[m,k] =-s*Rnk + c*R[m,k];

                QTnk = QT[n,k]
                QT[n,k] = c*QTnk + s*QT[m,k]
                QT[m,k] =-s*QTnk + c*QT[m,k];
            #print 'QT:\n',QT
            #print 'R:\n',R
            #print '-------------'

    return QT.T,R
            
def inv(in_A):
    """
    computes the inverse of A by
    
    STEP 1: QR decomposition
    STEP 2: Solution of the  extended linear system::
    
            (Q R | I) = ( R | QT )
            
            i.e.
            /R_11 R_12 R_13 ... R_1M | 1 0 0 0 ... 0 \
            | 0   R_22 R_23 ... R_2M | 0 1 0 0 ... 0 |
            | 0         ... ... .... |               |
            \                   R_NM | 0 0 0 0 ... 1 /
    
    
    """
    Q,R = qr(in_A)
    QT = Q.T
    N = shape(in_A)[0]
  
    for n in range(N-1,-1,-1):
        Rnn = R[n,n]
        R[n,:] /= Rnn
        QT[n,:] /= Rnn
        for m in range(n+1,N):
            Rnm = R[n,m]
            R[n,m] = 0
            QT[n,:] -= QT[m,:]*Rnm

    return QT 
