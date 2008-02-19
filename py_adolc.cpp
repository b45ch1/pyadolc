#include "py_adolc.hpp"

// template<class T>
// print_vec(T vec)


bpn::array wrapped_gradient(int tape_tag, bpn::array &x0){
	nu::check_rank(x0,1);
	vector<intp> shp(nu::shape(x0));
	int n = shp[0]; // lenght of x0
	
	/* SETUP VARIABLES */
	double* dataPtr = (double*) nu::data(x0);
	vector<double> g(n);

	/*call gradient function*/
	gradient(tape_tag,n,dataPtr,&g[0]);
	
	return nu::makeNum( &g[0], n);
}



bpn::array wrapped_function(int tape_tag, int codimension, bpn::array &x0){
	nu::check_rank(x0,1);
	vector<intp> shp(nu::shape(x0));
	int n = shp[0]; // lenght of x0

	/* SETUP VARIABLES */
	double* dataPtr = (double*) nu::data(x0);
	vector<double> y(codimension);
	
	function(tape_tag, codimension, n, dataPtr, &y[0]);
	return nu::makeNum( &y[0], codimension);
}
