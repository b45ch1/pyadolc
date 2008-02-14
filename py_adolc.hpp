#ifndef AP_HPP
#define AP_HPP
#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle



#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/scope.hpp>




#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include "num_util.h"

#include "adolc.h"
#include "adouble.h"

using namespace std;
namespace b = boost;
namespace bp = boost::python;
namespace bpn = boost::python::numeric;
namespace nu = num_util;


/*
class ADOLC_DLL_EXPORT pyadub:public adub{
public:
pyadub() : adub(){}
pyadub(double x) : adub(x){;}
pyadub(locint lo) : adub(lo){}
};*/






class mytestclass{
public:
int a;
mytestclass(){a = 1;}
};




std::ostream& operator<<(std::ostream& s, mytestclass const& x)
{
    return s << "x.value()";
}

template<class T>
void myprintf(T x){
	cout<<x<<endl;

}


int square(int number) { return number * number; }
double (*fmax_const_double_const_double)(const double &, const double &) = &fmax;

BOOST_PYTHON_MODULE(Adolc)
{
	using namespace boost::python;
	bp::scope().attr("__doc__") ="\
			my docstring \
			";
	import_array();
	bpn::array::set_module_and_type("numpy", "ndarray");

	/*simple test functions / classes that will be deleted later */
    bp::def("square",square);
	bp::def("fmax", fmax_const_double_const_double);
	bp::def("myprintf",myprintf<adouble>);
	
	bp::class_<mytestclass>("mytestclass")
		.def( boost::python::self_ns::str(self))
	;


	/* Wrapper for ADOLC */

	bp::def("trace_on",trace_on)
	
	
	bp::class_<adouble>("adouble", bp::init<double>())
			.def(bp::init<const adouble>())
			.def(bp::init<const adub>())
			.def(bp::self *= bp::self )

			/*lhs operators */
			.def(bp::self *= int() )
			.def(bp::self *= double() )
			.def(bp::self *  bp::self )
			.def(bp::self *  int() )
			.def(bp::self *  double() )
// 			.def(bp::self * adub())
			
			/*rhs operators */
			.def(double() *  bp::self )
			.def(int() *  bp::self )
			
			/* string representation */
			.def(boost::python::self_ns::str(self))
	;
// 	bp::class_<badouble>("badouble", bp::init<const badouble &>())
// 			.def(bp::init<adub>())
// 			/* string representation */
// 			.def(boost::python::self_ns::str(bp::self))
;

	bp::class_<adub>("adub",bp::init<locint>())
// 			.def(bp::init<locint>())
			.def(bp::self * adouble())
			.def(boost::python::self_ns::str(bp::self))
			
	;

/*	bp::class_<pyadub>("adub", bp::init<void>())*/
// 			.def(bp::init<const double>())
// 			.def(bp::init<locint>())
	;
}

#endif
