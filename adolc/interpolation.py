import numpy


def entangle_cross(V, V1, V2, V12):

    """
     using the interpolation formula

     v1.T H v2 + J v12 = 0.5 * ( u.T H u - v1.T H v1 - v2.T H v2 ) + J v12

     where u = v1 + v2
     and v1 columns of V1 and v2 columns of V2
    """


    N = V.shape[0]
    P = V.shape[1]
    D = V.shape[2]

    M = V1.shape[1]
    L = V2.shape[1]


    if(V.ndim != 3):
        raise ValueError("V.ndim = %d but expected V.ndim = 3 "%V.ndim)

    if(P != M + L + M*L):
        raise ValueError("The number of entangled directions P=%d of V " \
                            "does not match the dimensions M=%d and L=%d " \
                            "of V1 and V2"%(P, M, L))

    if(V12.shape[0] != N or V12.shape[1] != M or V12.shape[2] != L):
        raise ValueError("expected V12.shape = (%d, %d, %d) but got " \
                            "V12.shape = (%d, %d, %d) ",
                            (N, M, L,
                            V12.shape[0], V12.shape[1], V12.shape[2] ))


    # set V[:,0,:M] = V1

    for m in range(M):
        for n in range(N):
            V[n, m, 0] = V1[n,m]

    # set V[:,0,M:M+L] = V2

    for l in range(L):
        for n in range(N):
            V[n, l+M, 0] = V2[n,l]


    #  set V[:,0,M+L:] with mixed elements of V1 and V2

    for m in range(M):
        for l in range(L):
            for n in range(N):
                V[n, m+l*M + M + L, 0] = V1[n,m] + V2[n,l]

    #  V[:,1,M+L:] = V12

    for m in range(M):
        for l in range(L):
            for n in range(N):
                V[n, m + l*M + M + L, 1] = V12[n,l,m]


def detangle_cross(V, V1, V2, V12):

    N = V.shape[0]
    P = V.shape[1]
    D = V.shape[2]

    M = V1.shape[1]
    L = V2.shape[1]


    # V1 = V[:,0,:M]

    for m in range(M):
        for n in range(N):
            V1[n, m] = V[n, m, 0]

    # V1 = V[:,0,M:M+L]

    for l in range(L):
        for n in range(N):
            V2[n, l] = V[n, M+l, 0]


    # build V12 from mixed derivatives in
    # V[:,1,:M], V[:,1,M:M+L] and V[:,1,M+L:]

    for m in range(M):
        for l in range(L):
            for n in range(N):
                V12[n, l, m] = V[n,  M + L + m + l*M, 1] - V[n, m, 1] - V[n, M+l, 1]
