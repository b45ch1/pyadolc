#ifndef PY_COLPACK_HPP
#define PY_COLPACK_HPP

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

#include "num_util.h"
#include "adolc/adolc.h"
#include "adolc/sparse/sparsedrivers.h"

using namespace std;
namespace b = boost;
namespace bp = boost::python;
namespace bpn = boost::python::numpy;
namespace nu = num_util;

// extern int jac_pat(short,int,int,double*,unsigned int**,int*);
// extern void generate_seed_jac(int, int, unsigned int**, double***, int*, int);
// extern int sparse_jac(short, int , int, int, double*, int *, unsigned int **, unsigned int **, double **,int*);
// extern int hess_pat(short,int,double*,unsigned int**, int);
// extern void generate_seed_hess(int, unsigned int**, double***, int*, int);
// extern int sparse_hess(short, int , int, double*, int *, unsigned int **, unsigned int **, double **,int*);
// extern int bit_vector_propagation(short, int, int, double*, unsigned int**, int*);

bp::list	wrapped_jac_pat(short tape_tag, bpn::ndarray &bpn_x, bpn::ndarray &bpn_options);
// bp::list	wrapped_sparse_jac_no_repeat(short tape_tag, bpn::ndarray &bpn_x, bpn::ndarray &bpn_options);
// bp::list	wrapped_sparse_jac_repeat(short tape_tag, bpn::ndarray &bpn_x, npy_intp nnz, bpn::ndarray &bpn_rind, bpn::ndarray &bpn_cind, bpn::ndarray &bpn_values);

bp::list	wrapped_hess_pat(short tape_tag, bpn::ndarray &bpn_x, npy_intp option);
// bp::list	wrapped_sparse_hess_no_repeat(short tape_tag, bpn::ndarray &bpn_x, bpn::ndarray &bpn_options);
// bp::list	wrapped_sparse_hess_repeat(short tape_tag, bpn::ndarray &bpn_x, npy_intp nnz, bpn::ndarray &bpn_rind, bpn::ndarray &bpn_cind, bpn::ndarray &bpn_values);

#define NUMPY_IMPORT_ARRAY_RETVAL

BOOST_PYTHON_MODULE(_sparse)
{
	using namespace boost::python;
  bpn::initialize();
  import_array();
	def("jac_pat", 	             &wrapped_jac_pat);
	// def("sparse_jac_no_repeat",  &wrapped_sparse_jac_no_repeat);
	// def("sparse_jac_repeat",  &wrapped_sparse_jac_repeat);

	def("hess_pat", 	                     &wrapped_hess_pat);
	// def("sparse_hess_no_repeat", 	         &wrapped_sparse_hess_no_repeat);
	// def("sparse_hess_repeat", 	             &wrapped_sparse_hess_repeat);
	
}

#endif
