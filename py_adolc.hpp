#ifndef AP_HPP
#define AP_HPP
#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

#include "num_util.h"

using namespace std;
namespace b = boost;
namespace bp = boost::python;
namespace bpn = boost::python::numeric;
namespace nu = num_util;

/** \brief Returns a matrix M, where M_{nm} = dot(x_n, x_m)
 * \param inMat A (N,D) array in row major format. Each row represents a position vector.
 * \return Returns a (N,N) array M, where M_{nm} = dot(x_n, x_m) is the inner product between the n'th and m'th row of inMat. */
bpn::array outer_dot(bpn::array &inMat);

/** \brief Runs dense implementation of Affinity Propagation.
 * \param inSimilaritiesMatrix A similarity matrix as (N,N) array in row major format as it is returned by outer_dot().
 * \param maxit Maximal number of iterations.
 * \param convit Number of iterations the clustering solution does not change to proclaim convergence.
 * \param lam A damping factor that prevents oscillating solutions. Ranges between 0.5 and 1.
 * \return Returns a (N,N) array M, where M_{nm} = dot(x_n, x_m) is the inner product between the n'th and m'th row of inMat. */
bp::dict ap(bpn::array &inSimilaritiesMatrix, uint maxit, uint convit, double lam);

BOOST_PYTHON_MODULE(Adolc)
{
	bp::scope().attr("__doc__") ="\
			my docstring \
			";
	import_array();
	bpn::array::set_module_and_type("numpy", "ndarray");


	def("ap",ap, " description \n    ");
	def("outer_dot",outer_dot,
	     "Computes a distance matrix of squared Euclidean distances. \n\
		CALL: \n \
			outer_dot(x) \n \
		INPUTS:\n \
		x: numpy (N,D)-array. Each row is one vector of dimension D. \n \
		OUTPUTS: \n \
		d: numpy (N,N)-array. d[n,m] = ||x_n - x_m||^2 ");
}

#endif
