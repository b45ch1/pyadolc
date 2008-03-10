#ifndef AP_HPP
#define AP_HPP
#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

#include "num_util.h"
#include "adolc/adolc.h"

using namespace std;
namespace b = boost;
namespace bp = boost::python;
namespace bpn = boost::python::numeric;
namespace nu = num_util;

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
extern adub fmax ( const badouble&, const badouble& );
extern adub fmax ( double, const badouble& );
extern adub fmax ( const badouble&, double );
extern adub fmin ( const badouble&, const badouble& );
extern adub fmin ( double, const badouble& );
extern adub fmin ( const badouble&, double );
extern adub ldexp ( const badouble&, int );
// extern adub frexp ( const badouble&, int* );
// extern adub erf   ( const badouble& );

/* THIN WRAPPER FOR OVERLOADED FUNCTIONS */
void trace_on_default_argument(short tape_tag){ trace_on(tape_tag,0);}
void trace_off_default_argument(){ trace_off(0);}


/* PYTHONIC CALLS OF FUNCTIONS */
bpn::array	wrapped_function			(short tape_tag, bpn::array &bpn_x);
bpn::array	wrapped_gradient			(short tape_tag, bpn::array &bpn_x);
bpn::array	wrapped_hessian				(short tape_tag, bpn::array &bpn_x);
bpn::array	wrapped_jacobian			(short tape_tag, bpn::array &bpn_x);
bpn::array	wrapped_vec_jac				(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_u, bool repeat);
bpn::array	wrapped_jac_vec				(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_v);
bpn::array	wrapped_hess_vec			(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_v);
bpn::array	wrapped_lagra_hess_vec		(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_u, bpn::array &bpn_v);
void		wrapped_jac_solv			(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_b, int sparse, int mode);
bpn::array	wrapped_zos_forward			(short tape_tag, bpn::array &bpn_x, int keep);
bp::tuple	wrapped_fos_forward			(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_v, int keep);
bp::tuple 	wrapped_fov_forward			(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_V);
bp::tuple	wrapped_hos_forward			(short tape_tag, int D, bpn::array &bpn_x, bpn::array &bpn_V, int keep);
bp::tuple	wrapped_hov_forward			(short tape_tag, int D, bpn::array &bpn_x, bpn::array &bpn_V);
bpn::array wrapped_fos_reverse			(short tape_tag, bpn::array &bpn_u);
bpn::array wrapped_fov_reverse			(short tape_tag, bpn::array &bpn_U);
bpn::array wrapped_hos_reverse			(short tape_tag, int D, bpn::array &bpn_u);
bp::tuple wrapped_hov_reverse			(short tape_tag, int D, bpn::array &bpn_U);

/* C STYLE CALLS OF FUNCTIONS */
void c_wrapped_function			(short tape_tag, int M, int N, bpn::array &bpn_x, bpn::array &bpn_y );
void c_wrapped_gradient			(short tape_tag, int N, bpn::array &bpn_x, bpn::array &bpn_g);
void c_wrapped_hessian			(short tape_tag, int N, bpn::array &bpn_x, bpn::array &bpn_H);
void c_wrapped_jacobian			(short tape_tag, int M, int N, bpn::array &bpn_x, bpn::array &bpn_J);
void c_wrapped_vec_jac			(short tape_tag, int M, int N, bool repeat, bpn::array &bpn_x, bpn::array &bpn_u, bpn::array &bpn_z);
void c_wrapped_jac_vec			(short tape_tag, int M, int N, bpn::array &bpn_x, bpn::array &bpn_v, bpn::array &bpn_z);
void c_wrapped_hess_vec			(short tape_tag, int N, bpn::array &bpn_x, bpn::array &bpn_v, bpn::array &bpn_z);
void c_wrapped_lagra_hess_vec	(short tape_tag, int M, int N, bpn::array &bpn_x, bpn::array &bpn_v, bpn::array &bpn_u,bpn::array &bpn_h);
void c_wrapped_jac_solv			(short tape_tag, int N, bpn::array &bpn_x, bpn::array &bpn_b, int sparse, int mode);
void c_wrapped_zos_forward		(short tape_tag, int M, int N, int keep, bpn::array &bpn_x, bpn::array &bpn_y);
void c_wrapped_fos_forward		(short tape_tag, int M, int N, int keep, bpn::array &bpn_x, bpn::array &bpn_v, bpn::array &bpn_y, bpn::array &bpn_w);
void c_wrapped_fov_forward		(short tape_tag, int M, int N, int P, bpn::array &bpn_x, bpn::array &bpn_V, bpn::array &bpn_y, bpn::array &bpn_W);
void c_wrapped_hos_forward		(short tape_tag, int M, int N, int D, int keep, bpn::array &bpn_x, bpn::array &bpn_V, bpn::array &bpn_y, bpn::array &bpn_W);
void c_wrapped_hov_forward		(short tape_tag, int M, int N, int D, int P, bpn::array &bpn_x, bpn::array &bpn_V, bpn::array &bpn_y, bpn::array &bpn_W);

void c_wrapped_fos_reverse		(short tape_tag, int M, int N, bpn::array &bpn_u, bpn::array &bpn_z);
void c_wrapped_fov_reverse		(short tape_tag, int M, int N, int Q, bpn::array &bpn_U, bpn::array &bpn_Z);
void c_wrapped_hos_reverse		(short tape_tag, int M, int N, int D, bpn::array &bpn_u, bpn::array &bpn_Z);
void c_wrapped_hov_reverse		(short tape_tag, int M, int N, int D, int Q, bpn::array &bpn_U, bpn::array &bpn_Z, bpn::array &bpn_nz);


void py_tape_doc(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_y );
bpn::array wrapped_tapestats(short tape_tag);


/* of class badouble */
adub	(*exp_adub) 		( const badouble& ) = &exp;
adub	(*log_adub) 		( const badouble& ) = &log;
adub	(*sqrt_adub)		( const badouble& ) = &sqrt;
adub	(*sin_adub) 		( const badouble& ) = &sin;
adub	(*cos_adub) 		( const badouble& ) = &cos;
adub	(*tan_adub) 		( const badouble& ) = &tan;
adub	(*asin_adub)		( const badouble& ) = &asin;
adub	(*acos_adub)		( const badouble& ) = &acos;
adub	(*atan_adub)		( const badouble& ) = &atan;
adub	(*log10_adub)		( const badouble& ) = &log10;
adub	(*sinh_adub)		( const badouble& ) = &sinh;
adub	(*cosh_adub) 		( const badouble& ) = &cosh;
adub	(*tanh_adub) 		( const badouble& ) = &tanh;
// adub (*asinh_adub) 		( const badouble& ) = &asinh;
// adub (*acosh_adub) 		( const badouble& ) = &acosh;
// adub (*atanh_adub) 		( const badouble& ) = &atanh;
adub	(*fabs_adub) 		( const badouble& ) = &fabs;
adub	(*ceil_adub)		( const badouble& ) = &ceil;
adub	(*floor_adub) 		( const badouble& ) = &floor;

adub	(*pow_adub)			( const badouble&, double ) = &pow;
adouble	(*pow_adouble_badouble_badouble)( const badouble&, const badouble& ) = &pow;
adouble (*pow_adouble_double_badouble)( double, const badouble& ) = &pow;
adub	(*fmax_adub_badouble_badouble)		( const badouble&, const badouble& ) = &fmax;
adub	(*fmax_adub_double_badouble)		( double, const badouble& ) = &fmax;
adub	(*fmax_adub_badouble_double)		( const badouble&, double ) = &fmax;
adub	(*fmin_adub_badouble_badouble)		( const badouble&, const badouble& ) = &fmin;
adub	(*fmin_adub_double_badouble)		( double, const badouble& ) = &fmin;
adub	(*fmin_adub_badouble_double)		( const badouble&, double ) = &fmin;
adub	(*ldexp_adub) 		( const badouble&, int ) = &ldexp;
// adub (*frexp_adub) 		( const badouble&, int* ) = &frexp;
// adub (*erf_adub) 		( const badouble& ) = &erf;

/* WRAPPED OPERATORS */
adub *adub_add_badouble_badouble(const badouble &lhs, const badouble &rhs){	return new adub(operator+(lhs,rhs));}
adub *adub_sub_badouble_badouble(const badouble &lhs, const badouble &rhs){	return new adub(operator-(lhs,rhs));}
adub *adub_mul_badouble_badouble(const badouble &lhs, const badouble &rhs){	return new adub(operator*(lhs,rhs));}
adub *adub_div_badouble_badouble(const badouble &lhs, const badouble &rhs){	return new adub(operator/(lhs,rhs));}

adub *adub_add_badouble_double(const badouble &lhs,double rhs){	return new adub(operator+(lhs,rhs));}
adub *adub_sub_badouble_double(const badouble &lhs,double rhs){	return new adub(operator-(lhs,rhs));}
adub *adub_mul_badouble_double(const badouble &lhs,double rhs){	return new adub(operator*(lhs,rhs));}
adub *adub_div_badouble_double(const badouble &lhs,double rhs){	return new adub(operator/(lhs,rhs));}

adub *adub_add_double_badouble(const badouble &rhs,double lhs){	return new adub(operator+(lhs,rhs));}
adub *adub_sub_double_badouble(const badouble &rhs,double lhs){	return new adub(operator-(lhs,rhs));}
adub *adub_mul_double_badouble(const badouble &rhs,double lhs){	return new adub(operator*(lhs,rhs));}
adub *adub_div_double_badouble(const badouble &rhs,double lhs){	return new adub(operator/(lhs,rhs));}

double depends_on(badouble &a){
	double coval;
	a.operator>>=(coval);
	return coval;
}

badouble& (badouble::*operator_eq_double) ( double ) = &badouble::operator=;
badouble& (badouble::*operator_eq_badouble) ( const badouble& ) = &badouble::operator=;
badouble& (badouble::*operator_eq_adub) ( const adub& ) = &badouble::operator=;


BOOST_PYTHON_MODULE(adolc)
{
	using namespace boost::python;
	import_array(); 										/* some kind of hack to get numpy working */
	bpn::array::set_module_and_type("numpy", "ndarray");	/* some kind of hack to get numpy working */
	
	scope().attr("__doc__") =" Adolc: Algorithmic Differentiation Software \n\
	see http://www.math.tu-dresden.de/~adol-c/ for documentation of Adolc \n\
	http://mathematik.hu-berlin.de/~walter for more information and documentation of this Python extension\n\
	\n\
	return values are always numpy arrays!\n\
	\n\
	Example Session: \n\
	from numpy import * \n\
	from adolc import * \n\
	def vector_f(x): \n\
	\tV=vander(x) \n\
	\treturn dot(v,x) \n\
	\n\
	trace_on(0)\n\
	map(adouble.is_independent,ax,x)\n\
	ay = scalar_f(ax)\n\
	y = array(map(depends_on,ay)) \n\
	trace_off() \n\
	y2 = function(0,x) \n\
	g = gradient(0,x) \n\
	";

	def("trace_on",trace_on_default_argument , "trace_on(tape_tag):\nstart recording to the tape with index tape_tag\n");
	def("trace_off",trace_off_default_argument ,"turn off tracing\n");

	def("function", 		&wrapped_function, "f=function(tape_tag,x):\nevaluate the function f(x) recorded on tape with index tape_tag\n");
	def("gradient", 		&wrapped_gradient, "g=gradient(tape_tag,x):\nevaluate the gradient g = f'(x), f:R^N -> R\n");
	def("hessian",			&wrapped_hessian,  "H=hessian(tape_tag,x):\nevaluate the hessian H = f\"(x), f:R^N -> R\n");
	def("jacobian", 		&wrapped_jacobian, "J=jacobian(tape_tag,x):\nevaluate the jacobian J = F'(x), F:R^N -> R^M \n");
	def("vec_jac",			&wrapped_vec_jac,  "z=vec_jac(tape_tag,x,u):\nevaluate u^T F'(x), F:R^N -> R^M\n");
	def("jac_vec",			&wrapped_jac_vec,  "z=jac_vec(tape_tag,x,v):\nevaluate  F\"(x)v, F:R^N -> R^M\n");
	def("hess_vec",			&wrapped_hess_vec, "z=hess_vec(tape_tag,x,v):\nevaluate  f\"(x)v, f:R^N -> R\n");
	def("lagra_hess_vec", 	&wrapped_lagra_hess_vec,  "z=lagra_hess_vec(tape_tag,x,u,v):\nevaluate  F\"(x)v, F:R^N -> R^M\n");
	def("jac_solv",			&wrapped_jac_solv,  "(void*)lagra_hess_vec(tape_tag,x,b,sparse=0,mode=2):\nsolve F'(x) b_new - b = 0  , F:R^N -> R^M\n"); /* buggy ! */

	def("zos_forward",		&wrapped_zos_forward, 	"zero order scalar forward:\n"\
													"y = zos_forward(tape_tag, x, keep)\n" \
													"F:R^N -> R^M\n" \
													"x is N-vector, y is M-vector\n" \
													"keep = 1 prepares for fos_reverse or fov_reverse\n" \
													"");
	def("fos_forward",		&wrapped_fos_forward, 	"first order scalar forward:\n"\
													"(y,w) = fos_forward(tape_tag, x, v, keep) \n"\
													"F:R^N -> R^M\n"\
													"x is N-vector, y is M-vector\n"\
													"v is N-vector, direction \n"\
													"w is M-vector, directional derivative \n"\
													"keep = 1 prepares for fos_reverse or fov_reverse\n"\
													"keep = 2 prepares for hos_reverse or hov_reverse\n"\
													"");
	def("fov_forward",		&wrapped_fov_forward,	"first order vector forward:\n"\
													"(y,W) = fov_forward(tape_tag, x, V, keep) \n"\
													"F:R^N -> R^M\n"\
													"x is N-vector, y is M-vector\n"\
													"V is (N x P)-matrix. P directions \n"\
													"W is (M x P)-matrix. P directiona derivatives \n"\
													"keep = 1 prepares for fos_reverse or fov_reverse\n"\
													"keep = 2 prepares for hos_reverse or hov_reverse\n"\
													"");
	def("hos_forward",		&wrapped_hos_forward,	"higher order scalar forward:\n"\
													"(y,W) = hos_forward(tape_tag, D, x, V, keep) \n"\
													"F:R^N -> R^M\n"\
													"x is N-vector, y is M-vector\n"\
													"D is the order of the derivative\n"\
													"V is (N x D)-matrix \n"\
													"W is (M x D)-matrix \n"\
													"keep = 1 prepares for fos_reverse or fov_reverse\n"\
													"D+1 >= keep > 2 prepares for hos_reverse or hov_reverse\n"\
													"");
	def("hov_forward",		&wrapped_hov_forward,	"higher order vector forward:\n"\
													"(y,W) = hov_forward(tape_tag, D, x, V, keep) \n"\
													"F:R^N -> R^M\n"\
													"x is N-vector, y is M-vector\n"\
													"D is the order of the derivative\n"\
													"V is (N x P x D)-matrix, P directions \n"\
													"W is (M x P x D)-matrix, P directional derivatives \n"\
													"");
             
	def("fos_reverse",		&wrapped_fos_reverse,	"first order scalar reverse:\n"\
													"z = fos_reverse(tape_tag, x, u) \n"\
													"F:R^N -> R^M\n"\
													"x is N-vector, y is M-vector\n"\
													"u is M-vector, adjoint direction \n"\
													"z is N-vector, adjoint directional derivative z= u F'(x) \n"\
													"after calling zos_forward, fos_forward or hos_forward with keep = 1 \n"\
													"");
     
	def("fov_reverse",		&wrapped_fov_reverse,	"first order vector reverse:\n"\
													"Z = fov_reverse(tape_tag, x, u) \n"\
													"F:R^N -> R^M\n"\
													"x is N-vector, y is M-vector\n"\
													"U is (QxM)-matrix, Q adjoint directions \n"\
													"Z is (QxN)-matrix, adjoint directional derivative Z = U F'(x) \n"\
													"after calling zos_forward, fos_forward or hos_forward with keep = 1 \n"\
													"");
													
	def("hos_reverse",		&wrapped_hos_reverse,	"higher order scalar reverse:\n"\
													"Z = hos_reverse(tape_tag, D, x, u) \n"\
													"F:R^N -> R^M\n"\
													"x is N-vector, y is M-vector\n"\
													"D is the order of the derivative\n"\
													"u is M-vector, adjoint vector \n"\
													"Z is (N x D+1)-matrix, adjoint directional derivative Z = [u^T F'(x), u^T F\" v[:,0], ...] \n"\
													"after calling fos_forward or hos_forward with keep = D+1 \n"\
													"");
	def("hov_reverse", 		&wrapped_hov_reverse,	"higher order scalar reverse:\n"\
													"(Z,nz) = hov_reverse(tape_tag, x, D, U)\n"\
													"F:R^N -> R^M\n"\
													"x is N-vector, y is M-vector\n"\
													"D is the order of the derivative\n"\      
													"U is (Q x M)-matrix, Q adjoint directions \n"\
													"Z is (Q x N x D+1)-matrix, adjoint directional derivative Z = [U F'(x), U F\" v[:,0], ...] \n"\
													"nz is (Q x N)-matrix, information about the sparsity of Z:\n"\
													"0:trivial, 1:linear, 2:polynomial, 3:rational, 4:transcendental, 5:non-smooth\n"\
													"after calling fos_forward or hos_forward with keep = D+1 \n"\
													"");

	/* c style functions */
	def("function", 		&c_wrapped_function);
	def("gradient", 		&c_wrapped_gradient);
	def("hessian",			&c_wrapped_hessian);
	def("jacobian", 		&c_wrapped_jacobian);
	def("vec_jac",			&c_wrapped_vec_jac);
	def("jac_vec",			&c_wrapped_jac_vec);
	def("hess_vec",			&c_wrapped_hess_vec);
	def("lagra_hess_vec", 	&c_wrapped_lagra_hess_vec);
	def("jac_solv",			&c_wrapped_jac_solv); /* buggy ! */

	def("zos_forward",		&c_wrapped_zos_forward);
	def("fos_forward",		&c_wrapped_fos_forward);
	def("fov_forward",		&c_wrapped_fov_forward);
	def("hos_forward",		&c_wrapped_hos_forward);
	def("hov_forward",		&c_wrapped_hov_forward);

	def("fos_reverse",		&c_wrapped_fos_reverse);
	def("fov_reverse",		&c_wrapped_fov_reverse);
	def("hos_reverse",		&c_wrapped_hos_reverse);
	def("hov_reverse", 		&c_wrapped_hov_reverse);
		
	def("depends_on", 		&depends_on);
	def("tape_to_latex",	py_tape_doc,	"\n\ntape_to_latex(tape_tag,x,y)\n"\
											"F:R^N -> R^M\n"\
											"x is N-vector  y is M-vector\n\n"\
											"writes the tape to a file called tape_x.tex that can be compile with Latex\n\n"\
											"");


	class_<badouble>("badouble", init<const badouble &>())
			.def(boost::python::self_ns::str(self))

			.add_property("val", &badouble::value)
			.add_property("location", &badouble::loc)
			
			.def("is_independent", &badouble::operator<<=, return_internal_reference<>())
			.def("__ilshift__", operator_eq_double, return_internal_reference<>())
			.def("__ilshift__", operator_eq_badouble, return_internal_reference<>())
			.def("__ilshift__", operator_eq_adub, return_internal_reference<>())

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

			.def("__add__", adub_add_badouble_badouble, return_value_policy<manage_new_object>())
			.def("__sub__", adub_sub_badouble_badouble, return_value_policy<manage_new_object>())
			.def("__mul__", adub_mul_badouble_badouble, return_value_policy<manage_new_object>())
			.def("__div__", adub_div_badouble_badouble, return_value_policy<manage_new_object>())

			.def("__radd__", adub_add_double_badouble, return_value_policy<manage_new_object>())
			.def("__rsub__", adub_sub_double_badouble, return_value_policy<manage_new_object>())
			.def("__rmul__", adub_mul_double_badouble, return_value_policy<manage_new_object>())
			.def("__rdiv__", adub_div_double_badouble, return_value_policy<manage_new_object>())

			.def("__add__", adub_add_badouble_double, return_value_policy<manage_new_object>())
			.def("__sub__", adub_sub_badouble_double, return_value_policy<manage_new_object>())
			.def("__mul__", adub_mul_badouble_double, return_value_policy<manage_new_object>())
			.def("__div__", adub_div_badouble_double, return_value_policy<manage_new_object>())
                        
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
			.def("arcsin",asin_adub )
			.def("arccos",acos_adub )
			.def("arctan",atan_adub )
			.def("log10",log10_adub)
			.def("sinh",sinh_adub)
			.def("cosh",cosh_adub)
			.def("tanh",tanh_adub)
// 			.def("asinh",asinh_adub)
// 			.def("acosh",acosh_adub)
// 			.def("atanh",atanh_adub)

			.def("fabs",fabs_adub)
			.def("ceil", ceil_adub)
			.def("floor", floor_adub)
			.def("fmax", fmax_adub_badouble_badouble)
			.def("fmax", fmax_adub_double_badouble)
			.def("fmax", fmax_adub_badouble_double)
			.def("fmin", fmin_adub_badouble_badouble)
			.def("fmin", fmin_adub_double_badouble)
			.def("fmin", fmin_adub_badouble_double)
	;

	class_<adub, bases<badouble> >("adub",init<locint>())
			.def(init<const adub &>())
	;
	
	class_<adouble, bases<badouble> >("adouble", init<double>())
			.def(init<const adouble&>())
			.def(init<const adub&>())
	;
	
	class_<asub, bases<badouble> >("asub",init<locint,locint>())
	;

}

#endif
