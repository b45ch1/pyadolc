#ifndef AP_HPP
#define AP_HPP
#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

#include "num_util.h"
#include "adolc.h"

using namespace std;
namespace b = boost;
namespace bp = boost::python;
namespace bpn = boost::python::numeric;
namespace nu = num_util;


/*thin wrapper for overloaded functions */
void trace_on_default_argument(short tag){ trace_on(tag,0);}
void trace_off_default_argument(){ trace_off(0);}
bpn::array wrapped_gradient(int tape_tag, bpn::array &compute_at_x0);
bpn::array wrapped_function(int tape_tag, int codimension, bpn::array &compute_at_x0);

double depends_on(badouble &a){
	double coval;
	a.operator>>=(coval);
	return coval;
}

BOOST_PYTHON_MODULE(Adolc)
{
	using namespace boost::python;
	import_array(); 										/* some kind of hack to get numpy working */
	bpn::array::set_module_and_type("numpy", "ndarray");	/* some kind of hack to get numpy working */
	
	scope().attr("__doc__") ="\
			my docstring \
	";

	def("trace_on",trace_on);
	def("trace_on",trace_on_default_argument);
	def("trace_off",trace_off);
	def("trace_off",trace_off_default_argument);


	def("gradient", &wrapped_gradient);
	def("function", &wrapped_function);
	def("depends_on", &depends_on);

	
	class_<badouble>("badouble", init<const badouble &>())
			.def(boost::python::self_ns::str(self))

			.add_property("val", &badouble::value)
			
			.def("is_independent", &badouble::operator<<=, return_internal_reference<>())
			.def("__ilshift__", &badouble::operator<<=, return_internal_reference<>())
			.def("__irshift__", &badouble::operator>>=, return_internal_reference<>())

			.def(-self)
			.def(+self)
			.def(self += double() )
			.def(self -= double() )
			.def(self *= double() )
			.def(self /= double() )

			.def(self += self )
			.def(self -= self )
			.def(self *= self )
			.def(self /= self )

			.def(self +  double() )
			.def(self -  double() )
			.def(self *  double() )
			.def(self /  double() )

			.def(double() + self )
			.def(double() - self )
			.def(double() * self )
			.def(double() / self )


			.def(self +  self )
			.def(self -  self )
			.def(self *  self )
			.def(self /  self )
			
// 			.def(self +  int() )
// 			.def(self -  int() )
// 			.def(self *  int() )
// 			.def(self /  int() )
	;

	class_<adub, bases<badouble> >("adub",init<locint>())
	;
	
	class_<adouble, bases<badouble> >("adouble", init<double>())
			.def(init<const adouble>())
			.def(init<const adub>())
	;
	class_<asub, bases<badouble> >("asub",init<locint,locint>())
	;

}

#endif
