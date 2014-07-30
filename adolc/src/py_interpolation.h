#ifndef PY_INTERPOLATION_H
#define PY_INTERPOLATION_H

#include "num_util.h"

using namespace std;
namespace b = boost;
namespace bp = boost::python;
namespace bpn = boost::python::numeric;
namespace nu = num_util;

/**

entangle_cross(V, V1, V2, V12)

using the interpolation formula

v1.T H v2 + J v12 = 0.5 * ( u.T H u - v1.T H v1 - v2.T H v2 ) + J v12

where u = v1 + v2
      v1 columns of V1
      v2 columns of V2

Parameters
----------

V:      (OUTPUT) DArray
        V.shape = (N, D, P)

V1      (INPUT) DArray
        V1.shape = (N, M)
        Jacobian-vector product

V2      (INPUT) DArray
        V2.shape = (N, L)
        Jacobian-vector product

V12     (INPUT) DArray
        V12.shape = (N, M, L)

N = number of input arguments
D = 2 = order of the Taylor polynomial
P = M*L + M + L number of entangled directions
*/
void entangle_cross(bpn::array &V, bpn::array &V1, bpn::array &V2, bpn::array &V12);

/**

inverse function of entangle_cross

Parameters
----------

V:      (INPUT) DArray
        V.shape = (N, D, P)

V1      (OUTPUT) DArray
        V1.shape = (N, M)
        Jacobian-vector product

V2      (OUTPUT) DArray
        V2.shape = (N, L)
        Jacobian-vector product

V12     (OUTPUT) DArray
        V12.shape = (N, M, L)

        cross-derivative part of the Hessian


*/
void detangle_cross(bpn::array &V, bpn::array &V1, bpn::array &V2, bpn::array &V12);


#endif