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



bpn::array wrapped_function(short tape_tag, bpn::array &bpn_x){
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

bpn::array wrapped_gradient(short tape_tag, bpn::array &bpn_x){
	nu::check_rank(bpn_x,1);
	vector<intp> shp(nu::shape(bpn_x));
	int N = shp[0]; // lenght of x
	double* x = (double*) nu::data(bpn_x);
	double g[N];
	gradient(tape_tag, N, x, g);
	return nu::makeNum(g, N);
}

bpn::array wrapped_hessian(short tape_tag, bpn::array &bpn_x){
	nu::check_rank(bpn_x,1);
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	double* x = (double*) nu::data(bpn_x);
	double** H = myalloc2(N,N);
	hessian(tape_tag, N, x, H);
	vector<intp> H_shape(2);
	H_shape[0]=N;
	H_shape[1]=N;
	return nu::makeNum( H[0], H_shape);
}

bpn::array wrapped_jacobian(short tape_tag, bpn::array &bpn_x){
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

bpn::array wrapped_vec_jac(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_u, bool repeat){
	nu::check_rank(bpn_x,1);
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];
	double* x = (double*) nu::data(bpn_x);
	double* u = (double*) nu::data(bpn_u);
	double	z[N];
	vec_jac(tape_tag, M, N, repeat, x, u, z);
	return nu::makeNum( z, N);
}

bpn::array wrapped_jac_vec(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_v){
	nu::check_rank(bpn_x,1);
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];
	double* x = (double*) nu::data(bpn_x);
	double* v = (double*) nu::data(bpn_v);
	double	z[M];
	jac_vec(tape_tag, M, N, x, v, z);
	return nu::makeNum( z, M);
}

bpn::array wrapped_hess_vec			(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_v){
	nu::check_rank(bpn_x,1);
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	double* x = (double*) nu::data(bpn_x);
	double* v = (double*) nu::data(bpn_v);
	double	z[N];
	hess_vec(tape_tag, N, x, v, z);
	return nu::makeNum( z, N);
}


bpn::array wrapped_lagra_hess_vec	(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_u, bpn::array &bpn_v){
	nu::check_rank(bpn_x,1);
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];
	double* x = (double*) nu::data(bpn_x);
	double* v = (double*) nu::data(bpn_v);
	double* u = (double*) nu::data(bpn_u);
	double	z[N];
	lagra_hess_vec(tape_tag, M, N, x, v, u, z);
	return nu::makeNum( z, N);
}

void wrapped_jac_solv(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_b, int sparse, int mode){
	nu::check_rank(bpn_x,1);
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	double* x = (double*) nu::data(bpn_x);
	double* b = (double*) nu::data(bpn_b);

	jac_solv(tape_tag, N, x, b, sparse, mode);
}

bpn::array wrapped_zos_forward (short tape_tag, bpn::array &bpn_x, int keep){
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];

	double* x = (double*) nu::data(bpn_x);
	double y[M];

	zos_forward(tape_tag, M, N, keep, x, y);
	return nu::makeNum( y, M);
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
	
	bp::list retvals;
	retvals.append(ret_y);
	retvals.append(ret_directional_derivative);
	return bp::tuple(retvals);
}

bp::tuple wrapped_fov_forward			(short tape_tag, bpn::array &bpn_x, bpn::array &bpn_V){
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];
	int P = nu::shape(bpn_V)[1];

	double* x = (double*) nu::data(bpn_x);
	double* V_data = (double*) nu::data(bpn_V);
	double* V[N];
	for(int n = 0; n != N; ++n){
		V[n] = &V_data[P * n];
	}
	
	double y[M];
	double** Y = myalloc2(M,P);

	fov_forward(tape_tag, M, N, P, x, V, y, Y);
	vector<intp> Y_shp(2); Y_shp[0] = M; Y_shp[1]=P;
	bpn::array ret_y 	=  nu::makeNum( y, M);
	bpn::array ret_Y 	=  nu::makeNum( Y[0], Y_shp);

	bp::list retvals;
	retvals.append(ret_y);
	retvals.append(ret_Y);
	return bp::tuple(retvals);
}

bp::tuple wrapped_hos_forward			(short tape_tag, int order, bpn::array &bpn_x, bpn::array &bpn_V, int keep){
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];
	int D = nu::shape(bpn_V)[1];

	double* x = (double*) nu::data(bpn_x);
	double* V_data = (double*) nu::data(bpn_V);
	double* V[N];
	for(int n = 0; n != N; ++n){
		V[n] = &V_data[D * n];
	}
	
	double y[M];
	double** Y = myalloc2(M,D);

	hos_forward(tape_tag, M, N, D, keep, x, V, y, Y);
	vector<intp> Y_shp(2); Y_shp[0] = M; Y_shp[1]=D;
	bpn::array ret_y 	=  nu::makeNum( y, M);
	bpn::array ret_Y 	=  nu::makeNum( Y[0], Y_shp);

	bp::list retvals;
	retvals.append(ret_y);
	retvals.append(ret_Y);
	return bp::tuple(retvals);
}


bp::tuple wrapped_hov_forward	(short tape_tag, int D, bpn::array &bpn_x, bpn::array &bpn_V){
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];
	int P = nu::shape(bpn_V)[1];

	double* x = (double*) nu::data(bpn_x);
	double* V_data = (double*) nu::data(bpn_V);
	double** V[N];
	double* V1[N*P];
	for(int n = 0; n != N; ++n){
		V[n] = &V1[ n * P];
	}
	for( int np = 0; np != N*P; ++np){
		V1[np] = &V_data[np * D];
	}
	
	double y[M];
	double*** W = myalloc3(M,P,D);

	hov_forward(tape_tag, M, N, D, P, x, V, y, W);
	vector<intp> W_shp(3); W_shp[0] = M; W_shp[1]=P; W_shp[2] = D;
	bpn::array ret_y 	=  nu::makeNum( y, M);
	bpn::array ret_W 	=  nu::makeNum( W[0][0], W_shp);

	bp::list retvals;
	retvals.append(ret_y);
	retvals.append(ret_W);
	return bp::tuple(retvals);
}


bpn::array wrapped_fos_reverse	(short tape_tag, bpn::array &bpn_u){
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];

	double* u = (double*) nu::data(bpn_u);
	double z[N];

	fos_reverse(tape_tag, M, N, u, z);
	return  nu::makeNum( z, N);
}

bpn::array wrapped_fov_reverse			(short tape_tag, bpn::array &bpn_U){
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];
	int Q = nu::shape(bpn_U)[0];

	double* U_data = (double*) nu::data(bpn_U);
	double* U[Q];
	for(int q = 0; q != Q; ++q){
		U[q] = &U_data[M * q];
	}
	
	double** Z = myalloc2(Q,N);

	fov_reverse(tape_tag, M, N, Q, U, Z);
	vector<intp> Z_shp(2); Z_shp[0] = Q; Z_shp[1] = N;
	return  nu::makeNum( Z[0], Z_shp);
}

bpn::array wrapped_hos_reverse			(short tape_tag, int D, bpn::array &bpn_u){
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];

	double* u = (double*) nu::data(bpn_u);

	double** Z = myalloc2(N,D+1);

	hos_reverse(tape_tag, M, N, D, u, Z);
	vector<intp> Z_shp(2); Z_shp[0] = N; Z_shp[1]=D+1;
	return nu::makeNum( Z[0], Z_shp);
}

bp::tuple wrapped_hov_reverse(short tape_tag, int D, bpn::array &bpn_U){
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	int N = tape_stats[NUM_INDEPENDENTS];
	int M = tape_stats[NUM_DEPENDENTS];
	int Q = nu::shape(bpn_U)[0];

	double* U_data = (double*) nu::data(bpn_U);
	double* U[Q];
	for(int q = 0; q != Q; ++q){
		U[q] = &U_data[M * q];
	}
	
	double*** Z = myalloc3(Q,N,D+1);
	short nz_data[Q*N];
	short* nz[Q];
	for(int q = 0; q != Q; ++q){
		nz[q] = &nz_data[q*N];
	}
	
	hov_reverse(tape_tag, M, N, D, Q, U, Z, nz);
	
	vector<intp> Z_shp(3); Z_shp[0] = Q; Z_shp[1] = N;  Z_shp[2]=D+1;
	vector<intp> nz_shp(2); nz_shp[0] = Q; nz_shp[1]=N;


	bpn::array ret_Z = nu::makeNum( Z[0][0], Z_shp);
	bpn::array ret_nz = nu::makeNum( nz[0], nz_shp);
	bp::list retvals;
	retvals.append(ret_Z);
	retvals.append(ret_nz);
	return bp::tuple(retvals);
	
}


/* C STYLE CALLS OF FUNCTIONS */
/* easy to use drivers */
void c_wrapped_function			(short tape_tag, int M, int N, bpn::array &bpn_x, bpn::array &bpn_y ){
	double* x = (double*) nu::data(bpn_x);
	double* y = (double*) nu::data(bpn_y);
	function(tape_tag, M, N, x, y);
}
void c_wrapped_gradient			(short tape_tag, int N, bpn::array &bpn_x, bpn::array &bpn_g){
	double* x = (double*) nu::data(bpn_x);
	double* g = (double*) nu::data(bpn_g);
	gradient(tape_tag, N, x, g);
}
void c_wrapped_hessian			(short tape_tag, int N, bpn::array &bpn_x, bpn::array &bpn_H){
	double* x = (double*) nu::data(bpn_x);
	double* H_data = (double*) nu::data(bpn_H);
	double* H[N];
	for(int n = 0; n != N; ++n){
		H[n] = &H_data[N * n];
	}
	hessian(tape_tag, N, x, H);
}
void c_wrapped_jacobian			(short tape_tag, int M, int N, bpn::array &bpn_x, bpn::array &bpn_J){
	double* x = (double*) nu::data(bpn_x);
	double* J[M];
	double* J_data = (double*) nu::data(bpn_J);
	for(int m = 0; m != M; ++m){
		J[m] = &J_data[N * m];
	}
	jacobian(tape_tag, M, N, x, J);
}
void c_wrapped_vec_jac			(short tape_tag, int M, int N, bool repeat, bpn::array &bpn_x, bpn::array &bpn_u, bpn::array &bpn_z){
	double* x = (double*) nu::data(bpn_x);
	double* u = (double*) nu::data(bpn_u);
	double* z = (double*) nu::data(bpn_z);
	vec_jac(tape_tag, M, N, repeat, x, u, z);
}
void c_wrapped_jac_vec			(short tape_tag, int M, int N, bpn::array &bpn_x, bpn::array &bpn_v, bpn::array &bpn_z){
	double* x = (double*) nu::data(bpn_x);
	double* v = (double*) nu::data(bpn_v);
	double* z = (double*) nu::data(bpn_z);
	jac_vec(tape_tag, M, N, x, v, z);
}
void c_wrapped_hess_vec			(short tape_tag, int N, bpn::array &bpn_x, bpn::array &bpn_v, bpn::array &bpn_z){
	double* x = (double*) nu::data(bpn_x);
	double* v = (double*) nu::data(bpn_v);
	double* z = (double*) nu::data(bpn_z);
	hess_vec(tape_tag, N, x, v, z);
}
void c_wrapped_lagra_hess_vec	(short tape_tag, int M, int N, bpn::array &bpn_x, bpn::array &bpn_v, bpn::array &bpn_u,bpn::array &bpn_h){

	double* x = (double*) nu::data(bpn_x);
	double* v = (double*) nu::data(bpn_v);
	double* u = (double*) nu::data(bpn_u);
	double* h = (double*) nu::data(bpn_h);
	lagra_hess_vec(tape_tag, M, N, x, v, u, h);
}
void c_wrapped_jac_solv			(short tape_tag, int N, bpn::array &bpn_x, bpn::array &bpn_b, int sparse, int mode){
	double* x = (double*) nu::data(bpn_x);
	double* b = (double*) nu::data(bpn_b);
	jac_solv(tape_tag, N, x, b, sparse, mode);
}

/* low level functions */
void c_wrapped_zos_forward		(short tape_tag, int M, int N, int keep, bpn::array &bpn_x, bpn::array &bpn_y){
	double* x = (double*) nu::data(bpn_x);
	double* y = (double*) nu::data(bpn_y);
	zos_forward(tape_tag, M, N, keep, x, y);
}
void c_wrapped_fos_forward		(short tape_tag, int M, int N, int keep, bpn::array &bpn_x, bpn::array &bpn_v, bpn::array &bpn_y, bpn::array &bpn_w){

	double* x = (double*) nu::data(bpn_x);
	double* v = (double*) nu::data(bpn_v);
	double* y = (double*) nu::data(bpn_y);
	double* w = (double*) nu::data(bpn_w);

	fos_forward(tape_tag, M, N, keep, x, v, y, w);
}

void c_wrapped_fov_forward		(short tape_tag, int M, int N, int P, bpn::array &bpn_x, bpn::array &bpn_V, bpn::array &bpn_y, bpn::array &bpn_W){
	double* x = (double*) nu::data(bpn_x);
	double* y = (double*) nu::data(bpn_y);
	double* V_data = (double*) nu::data(bpn_V);
	double* V[N];
	for(int n = 0; n != N; ++n){
		V[n] = &V_data[P * n];
	}
	double* W_data = (double*) nu::data(bpn_W);
	double* W[M];
	for(int m = 0; m != M; ++m){
		W[m] = &W_data[P * m];
	}

	fov_forward(tape_tag, M, N, P, x, V, y, W);
}

void c_wrapped_hos_forward		(short tape_tag, int M, int N, int D, int keep, bpn::array &bpn_x, bpn::array &bpn_V, bpn::array &bpn_y, bpn::array &bpn_W){
	double* x = (double*) nu::data(bpn_x);
	double* y = (double*) nu::data(bpn_y);
	double* V_data = (double*) nu::data(bpn_V);
	double* V[N];
	for(int n = 0; n != N; ++n){
		V[n] = &V_data[D * n];
	}
	double* W_data = (double*) nu::data(bpn_W);
	double* W[M];
	for(int m = 0; m != M; ++m){
		W[m] = &W_data[D * m];
	}
	hos_forward(tape_tag, M, N, D, keep, x, V, y, W);
}

void c_wrapped_hov_forward		(short tape_tag, int M, int N, int D, int P, bpn::array &bpn_x, bpn::array &bpn_V, bpn::array &bpn_y, bpn::array &bpn_W){
	double* x = (double*) nu::data(bpn_x);
	double* y = (double*) nu::data(bpn_y);
	double* V_data = (double*) nu::data(bpn_V);
	double** V[N];
	double* V1[N*P];
	for(int n = 0; n != N; ++n){
		V[n] = &V1[ n * P];
	}
	for( int np = 0; np != N*P; ++np){
		V1[np] = &V_data[np * D];
	}
	double* W_data = (double*) nu::data(bpn_W);
	double** W[M];
	double* W1[M*P];
	for(int m = 0; m != M; ++m){
		W[m] = &W1[ m * P];
	}
	for( int np = 0; np != N*P; ++np){
		W1[np] = &W_data[np * D];
	}
	hov_forward(tape_tag, M, N, D, P, x, V, y, W);
}

void c_wrapped_fos_reverse		(short tape_tag, int M, int N, bpn::array &bpn_u, bpn::array &bpn_z){
	double* u = (double*) nu::data(bpn_u);
	double* z = (double*) nu::data(bpn_z);
	fos_reverse(tape_tag, M, N, u, z);
}

void c_wrapped_fov_reverse		(short tape_tag, int M, int N, int Q, bpn::array &bpn_U, bpn::array &bpn_Z){
	double* U_data = (double*) nu::data(bpn_U);
	double* U[Q];
	for(int q = 0; q != Q; ++q){
		U[q] = &U_data[M * q];
	}
	
	double* Z_data = (double*) nu::data(bpn_Z);
	double* Z[Q];
	for(int q = 0; q != Q; ++q){
		Z[q] = &Z_data[N * q];
	}
	fov_reverse(tape_tag, M, N, Q, U, Z);
}
void c_wrapped_hos_reverse		(short tape_tag, int M, int N, int D, bpn::array &bpn_u, bpn::array &bpn_Z){
	double* u = (double*) nu::data(bpn_u);
	double* Z_data = (double*) nu::data(bpn_Z);
	double* Z[N];
	for(int n = 0; n != N; ++n){
		Z[n] = &Z_data[ n * (D+1)];
	}
	hos_reverse(tape_tag, M, N, D, u, Z);
}
void c_wrapped_hov_reverse		(short tape_tag, int M, int N, int D, int Q, bpn::array &bpn_U, bpn::array &bpn_Z, bpn::array &bpn_nz){
	double* U_data = (double*) nu::data(bpn_U);
	double* U[Q];
	for(int q = 0; q != Q; ++q){
		U[q] = &U_data[M * q];
	}

	double* Z_data = (double*) nu::data(bpn_Z);
	double** Z[Q];
	double* Z1[Q*N];
	for(int q = 0; q != Q; ++q){
		Z[q] = &Z1[ q * N];
	}
	for( int qn = 0; qn != Q*N; ++qn){
		Z1[qn] = &Z_data[qn * (D+1)];
	}

	short* nz_data = (short*) nu::data(bpn_nz);
	short* nz[Q];
	for(int q = 0; q != Q; ++q){
		nz[q] = &nz_data[ q * N];
	}
	hov_reverse(tape_tag, M, N, D, Q, U, Z, nz);
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


