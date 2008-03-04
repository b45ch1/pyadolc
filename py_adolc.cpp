#include <iostream>
#include <string>
#include "py_adolc.hpp"


using namespace std;
template<class T>
void print_vec(vector<T> vec, string msg=""){
	printf("%s",msg.c_str());
	printf("[ ");
	for(int i = 0; i!= vec.size(); ++i){
		cout<<vec[i]<<" ";
	}
	printf("]\n");
}

void print_array(double *vec, int length, string msg=""){
	printf("%s",msg.c_str());
	printf("array of length %d: [ ",length);
	for(int i = 0; i!= length; ++i){
		printf("%0.5f ",vec[i]);
	}
	printf("]\n");
}

bpn::array wrapped_gradient(uint tape_tag, bpn::array &bpn_x){
	nu::check_rank(bpn_x,1);
	vector<intp> shp(nu::shape(bpn_x));
	int N = shp[0]; // lenght of x
	double* x = (double*) nu::data(bpn_x);
	double g[N];
	gradient(tape_tag, N, x, g);
	return nu::makeNum(g, N);
}

bpn::array wrapped_jacobian(int tape_tag, bpn::array &bpn_x){
	nu::check_rank(bpn_x,1);
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];
	vector<intp> shp(nu::shape(bpn_x));
	if( N != shp[0]) cout<<"shape missmatch between tape and input vector (function wrapped_jacobian)"<<endl;
	double* x = (double*) nu::data(bpn_x);
	double** J = myalloc2(M,N);
	jacobian(tape_tag, M, N, x, J);
	vector<intp> J_shape(2);
	J_shape[0]=M;
	J_shape[1]=N;
	return nu::makeNum( J[0], J_shape);
}

bpn::array wrapped_function(int tape_tag, bpn::array &bpn_x){
	if(!nu::iscontiguous(bpn_x)){
		printf("not a contiguous array!\n");
	}
	nu::check_rank(bpn_x,1);
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];
	double* x = (double*) nu::data(bpn_x);
	vector<double> y(M);
	
	function(tape_tag, M, N, x, &y[0]);
	return nu::makeNum( &y[0], M);
}

bp::tuple wrapped_fos_forward(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_v, int keep){
	nu::check_rank(bpn_x,1);
	nu::check_rank(bpn_v,1);
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];

	double* x = (double*) nu::data(bpn_x);
	double* v = (double*) nu::data(bpn_v);
	vector<double> y(M);
	vector<double> directional_derivative(M);

	fos_forward(tape_tag, M, N, keep, x, v, &y[0], &directional_derivative[0]);

	bpn::array ret_y 	=  nu::makeNum( &y[0], M);
	bpn::array ret_directional_derivative 	=  nu::makeNum( &directional_derivative[0], M);
	
	bp::tuple retvals;
	retvals[0] = ret_y;
	retvals[1] = ret_directional_derivative;
	return retvals;
}


void py_tape_doc(short tape_tag, bpn::array &x, bpn::array &y ){
	nu::check_rank(x,1);
	nu::check_rank(y,1);

	double* dataPtr_x = (double*) nu::data(x);
	double* dataPtr_y = (double*) nu::data(y);
	int n = nu::shape(x)[0];
	int m = nu::shape(y)[0];

	tape_doc(tape_tag, m , n, dataPtr_x, dataPtr_y);


}

/* from taping.h and taping.c */
bpn::array wrapped_tapestats(short tape_tag) {
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	return nu::makeNum( tape_stats, STAT_SIZE);
}


