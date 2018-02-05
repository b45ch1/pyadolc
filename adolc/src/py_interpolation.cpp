#include "py_interpolation.h"


static inline int _(const int I, const int J, const int i, const int j){
    return J*i + j;
}

static inline int _(const int I, const int J, const int K,
                    const int i, const int j, const int k){
    return J*K*i + K*j +k;
}

void entangle_cross(bpn::ndarray &bpn_V, bpn::ndarray &bpn_V1, bpn::ndarray &bpn_V2, bpn::ndarray &bpn_V12){

    double* V   = (double*) nu::data(bpn_V);
    double* V1  = (double*) nu::data(bpn_V1);
    double* V2  = (double*) nu::data(bpn_V2);
    double* V12 = (double*) nu::data(bpn_V12);

    int N = nu::shape(bpn_V)[0];
    int P = nu::shape(bpn_V)[1];
    int D = nu::shape(bpn_V)[2];
    int M = nu::shape(bpn_V1)[1];
    int L = nu::shape(bpn_V2)[1];

    // # set V[:,0,:M] = V1
    for(int n=0; n < N; ++n){
        for(int m=0; m < M; ++m){
            V[_(N, P, D, n, m, 0)] = V1[_(N, M, n, m)];
        }
    }

    // # set V[:,0,M:M+L] = V2

    for(int n=0; n < N; ++n){
        for(int l=0; l < L; ++l){
            V[_(N, P, D, n, l+M, 0)] = V2[_(N, L, n,l)];
        }
    }

    for(int n=0; n < N; ++n){
        for(int l=0; l < L; ++l){
            V[_(N, P, D, n, l+M, 0)] = V2[_(N, L, n,l)];
        }
    }

    // #  set V[:,0,M+L:] with mixed elements of V1 and V2

    for(int n=0; n < N; ++n){
        for(int m=0; m < M; ++m){
            for(int l=0; l < L; ++l){
                V[_(N, P, D, n, m+l*M + M + L, 0)] = V1[_(N, M, n, m)] + V2[_(N, L, n,l)];
            }
        }
    }

    for(int n=0; n < N; ++n){
        for(int m=0; m < M; ++m){
            for(int l=0; l < L; ++l){
                V[_(N, P, D, n, m+l*M + M + L, 0)] = V1[_(N, M, n,m)] + V2[_(N, L, n,l)];
            }
        }
    }

    // #  V[:,1,M+L:] = V12
    for(int n=0; n < N; ++n){
        for(int m=0; m < M; ++m){
            for(int l=0; l < L; ++l){
                V[_(N, P, D, n, m + l*M + M + L, 1)] = V12[_(N, M, L, n, m, l)];
            }
        }
    }
}


void detangle_cross(bpn::ndarray &bpn_V, bpn::ndarray &bpn_V1, bpn::ndarray &bpn_V2, bpn::ndarray &bpn_V12){

    double* V   = (double*) nu::data(bpn_V);
    double* V1  = (double*) nu::data(bpn_V1);
    double* V2  = (double*) nu::data(bpn_V2);
    double* V12 = (double*) nu::data(bpn_V12);

    int N = nu::shape(bpn_V)[0];
    int P = nu::shape(bpn_V)[1];
    int D = nu::shape(bpn_V)[2];
    int M = nu::shape(bpn_V1)[1];
    int L = nu::shape(bpn_V2)[1];

    // # V1 = V[:,0,:M]

    for(int n=0; n < N; ++n){
        for(int m=0; m < M; ++m){
            V1[_(N, M, n, m)] = V[_(N, P, D, n, m, 0)];
        }
    }

    // # V1 = V[:,0,M:M+L]
    for(int n=0; n < N; ++n){
        for(int l=0; l < L; ++l){
            V2[_(N, L, n, l)] = V[_(N, P, D, n, M+l, 0)];
        }
    }

    // # build V12 from mixed derivatives in
    // # V[:,1,:M], V[:,1,M:M+L] and V[:,1,M+L:]

    for(int n=0; n < N; ++n){
        for(int m=0; m < M; ++m){
            for(int l=0; l < L; ++l){
                V12[_(N, M, L, n, m, l)] = V[_(N, P, D, n,  M + L + m + l*M, 1)] 
                                         - V[_(N, P, D, n, m, 1)] - V[_(N, P, D, n, M+l, 1)];
            }
        }
    }
}
