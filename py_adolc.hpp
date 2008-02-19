#ifndef AP_HPP
#define AP_HPP
#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

#include "num_util.h"
#include "adolc.h"
#include "adouble.h"

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

extern adub exp		( const badouble& );
extern adub log		( const badouble& );
extern adub sqrt	( const badouble& );
extern adub sin		( const badouble& );
extern adub cos		( const badouble& );
extern adub tan		( const badouble& );
extern adub asin	( const badouble& );
extern adub acos	( const badouble& );
extern adub atan	( const badouble& );
extern adub pow		( const badouble&, double );
extern adouble pow	( const badouble&, const badouble& );
extern adouble pow	( double, const badouble& ); /*this one doesnt work correctly yet */

extern adub log10	( const badouble& );
extern adub sinh  ( const badouble& );
extern adub cosh  ( const badouble& );
extern adub tanh  ( const badouble& );
// extern adub asinh ( const badouble& );
// extern adub acosh ( const badouble& );
// extern adub atanh ( const badouble& );
extern adub fabs  ( const badouble& );
extern adub ceil  ( const badouble& );
extern adub floor ( const badouble& );
// extern adub fmax ( const badouble&, const badouble& );
// extern adub fmax ( double, const badouble& );
// extern adub fmax ( const badouble&, double );
// extern adub fmin ( const badouble&, const badouble& );
// extern adub fmin ( double, const badouble& );
// extern adub fmin ( const badouble&, double );
// extern adub ldexp ( const badouble&, int );
// extern adub frexp ( const badouble&, int* );
// extern adub erf   ( const badouble& );



adub (*exp_adub) 		( const badouble& ) = &exp;
adub (*log_adub) 		( const badouble& ) = &log;
adub (*sqrt_adub)		( const badouble& ) = &sqrt;
adub (*sin_adub) 		( const badouble& ) = &sin;
adub (*cos_adub) 		( const badouble& ) = &cos;
adub (*tan_adub) 		( const badouble& ) = &tan;
adub (*asin_adub)		( const badouble& ) = &asin;
adub (*acos_adub)		( const badouble& ) = &acos;
adub (*atan_adub)		( const badouble& ) = &atan;
adub (*pow_adub)		( const badouble&, double ) = &pow;
adouble (*pow_adouble_badouble_badouble)( const badouble&, const badouble& ) = &pow;
adouble (*pow_adouble_double_badouble)( double, const badouble& ) = &pow;

adub (*log10_adub)		( const badouble& ) = &log10;
adub (*sinh_adub)		( const badouble& ) = &sinh;
adub (*cosh_adub) 		( const badouble& ) = &cosh;
adub (*tanh_adub) 		( const badouble& ) = &tanh;
// adub (*asinh_adub) 		( const badouble& ) = &asinh;
// adub (*acosh_adub) 		( const badouble& ) = &acosh;
// adub (*atanh_adub) 		( const badouble& ) = &atanh;

adub (*fabs_adub) 		( const badouble& ) = &fabs;
adub (*ceil_adub)		( const badouble& ) = &ceil;
adub (*floor_adub) 		( const badouble& ) = &floor;
// 
// adub (*fmax_adub_badouble_badouble)		( const badouble&, const badouble& ) = &fmax;
// adub (*fmax_adub_double_badouble)		( double, const badouble& ) = &fmax;
// adub (*fmax_adub_badouble_double)		( const badouble&, double ) = &fmax;
// adub (*fmin_adub_badouble_badouble)		( const badouble&, const badouble& ) = &fmin;
// adub (*fmin_adub_double_badouble)		( double, const badouble& ) = &fmin;
// adub (*fmin_adub_badouble_double)		( const badouble&, double ) = &fmin;
// 
// adub (*ldexp_adub) 		( const badouble&, int ) = &ldexp;
// adub (*frexp_adub) 		( const badouble&, int* ) = &frexp;
// adub (*erf_adub) 		( const badouble& ) = &erf;



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

			.def("__pow__",pow_adub)
			.def("__pow__",pow_adouble_badouble_badouble)
			.def("__rpow__",pow_adouble_double_badouble)

			.def("exp", exp_adub  )
			.def("log", log_adub  )
			.def("sqrt",sqrt_adub )
			.def("sin", sin_adub  )
			.def("cos", cos_adub  )
			.def("tan", tan_adub  )
			.def("asin",asin_adub )
			.def("acos",acos_adub )
			.def("atan",atan_adub )
			.def("log10",log10_adub)
/*			.def("sinh",sinh_adub)
			.def("cosh",cosh_adub)
			.def("tanh",tanh_adub)
			.def("asinh",asinh_adub)
			.def("acosh",acosh_adub)
			.def("atanh",atanh_adub)	*/		
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
