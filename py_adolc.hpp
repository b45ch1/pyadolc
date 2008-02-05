#ifndef AP_HPP
#define AP_HPP
#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include "num_util.h"

#include "adolc.h"

using namespace std;
namespace b = boost;
namespace bp = boost::python;
namespace bpn = boost::python::numeric;
namespace nu = num_util;


int square(int number) { return number * number; }
double (*fmax_const_double_const_double)(const double &, const double &) = &fmax;

BOOST_PYTHON_MODULE(Adolc)
{
	bp::scope().attr("__doc__") ="\
			my docstring \
			";
	import_array();
	bpn::array::set_module_and_type("numpy", "ndarray");
    	bp::def("square",square);
	bp::def("fmax", fmax_const_double_const_double);
	
	bp::class_<adouble>("adouble", bp::init<double>())
			.def(bp::init<const adouble>())
			.def(bp::init<const adub>())
			.def(bp::self *= bp::self )
			.def(bp::self *  bp::self )
	;
	
}

#endif
