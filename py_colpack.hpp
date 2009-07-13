#ifndef PY_COLPACK_HPP
#define PY_COLPACK_HPP

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

#include "num_util.h"
#include "adolc/adolc.h"
#include "adolc/sparse/sparsedrivers.h"

using namespace std;
namespace b = boost;
namespace bp = boost::python;
namespace bpn = boost::python::numeric;
namespace nu = num_util;

// extern int jac_pat(short,int,int,double*,unsigned int**,int*);
// extern void generate_seed_jac(int, int, unsigned int**, double***, int*, int);
// extern int sparse_jac(short, int , int, int, double*, int *, unsigned int **, unsigned int **, double **,int*);
// extern int hess_pat(short,int,double*,unsigned int**, int);
// extern void generate_seed_hess(int, unsigned int**, double***, int*, int);
// extern int sparse_hess(short, int , int, double*, int *, unsigned int **, unsigned int **, double **,int*);
// extern int bit_vector_propagation(short, int, int, double*, unsigned int**, int*);




bp::list	wrapped_jac_pat(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_options);
bp::list	wrapped_sparse_jac_no_repeat(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_options);
bp::list	wrapped_sparse_jac_repeat(short tape_tag, bpn::array &bpn_x, npy_intp nnz, bpn::array &bpn_rind, bpn::array &bpn_cind, bpn::array &bpn_values);

bp::list	wrapped_hess_pat(short tape_tag, bpn::array &bpn_x, npy_intp option);
bp::list	wrapped_sparse_hess_no_repeat(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_options);
bp::list	wrapped_sparse_hess_repeat(short tape_tag, bpn::array &bpn_x, npy_intp nnz, bpn::array &bpn_rind, bpn::array &bpn_cind, bpn::array &bpn_values);






BOOST_PYTHON_MODULE(_colpack)
{
	using namespace boost::python;
	import_array(); 										/* some kind of hack to get numpy working */
	bpn::array::set_module_and_type("numpy", "ndarray");	/* some kind of hack to get numpy working */
	def("jac_pat", 	             &wrapped_jac_pat);
	def("sparse_jac_no_repeat",  &wrapped_sparse_jac_no_repeat);
	def("sparse_jac_repeat",  &wrapped_sparse_jac_repeat);

	def("hess_pat", 	                     &wrapped_hess_pat);
	def("sparse_hess_no_repeat", 	         &wrapped_sparse_hess_no_repeat);
	def("sparse_hess_repeat", 	             &wrapped_sparse_hess_repeat);
	
}

#endif
