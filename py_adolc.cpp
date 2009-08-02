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


bp::dict wrapped_tapestats(short tape_tag){
	int tape_stats[STAT_SIZE];
	tapestats(tape_tag, tape_stats);
	bp::dict retval;
	retval["NUM_INDEPENDENTS"] = tape_stats[NUM_INDEPENDENTS];
	retval["NUM_DEPENDENTS"]   = tape_stats[NUM_DEPENDENTS];
	retval["NUM_MAX_LIVES"]    = tape_stats[NUM_MAX_LIVES];
	retval["TAY_STACK_SIZE"] = tape_stats[TAY_STACK_SIZE];
	retval["OP_BUFFER_SIZE"] = tape_stats[OP_BUFFER_SIZE];
	retval["NUM_OPERATIONS"] = tape_stats[NUM_OPERATIONS];
	retval["OP_FILE_ACCESS"] = tape_stats[OP_FILE_ACCESS];
	retval["NUM_LOCATIONS"] = tape_stats[NUM_LOCATIONS];
	retval["LOC_FILE_ACCESS"] = tape_stats[LOC_FILE_ACCESS];
	retval["NUM_VALUES"] = tape_stats[NUM_VALUES];
	retval["VAL_FILE_ACCESS"] = tape_stats[VAL_FILE_ACCESS];
	retval["LOC_BUFFER_SIZE"] = tape_stats[LOC_BUFFER_SIZE];
	retval["VAL_BUFFER_SIZE"] = tape_stats[VAL_BUFFER_SIZE];
	retval["TAY_BUFFER_SIZE"] = tape_stats[TAY_BUFFER_SIZE];
	return retval;
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
// void c_wrapped_jac_solv			(short tape_tag, int N, bpn::array &bpn_x, bpn::array &bpn_b, int sparse, int mode){
// 	double* x = (double*) nu::data(bpn_x);
// 	double* b = (double*) nu::data(bpn_b);
// 	jac_solv(tape_tag, N, x, b, sparse, mode);
// }

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

	for( int np = 0; np != N*P; ++np){
		V1[np] = &V_data[np * D];
	}
	for(int n = 0; n != N; ++n){
		V[n] = &V1[ n * P];
	}
	
	double* W_data = (double*) nu::data(bpn_W);
	double** W[M];
	double* W1[M*P];
	for( int mp = 0; mp != M*P; ++mp){
		W1[mp] = &W_data[mp * D];
	}
	for(int m = 0; m != M; ++m){
		W[m] = &W1[ m * P];
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


void c_wrapped_hov_ti_reverse	(short tape_tag, int M, int N, int D, int Q, bpn::array &bpn_U, bpn::array &bpn_Z, bpn::array &bpn_nz){

	/* U is (Q, M, D+1) matrix */
	double* U_data = (double*) nu::data(bpn_U);
	double** U[Q];
	double* U1[Q*M];
	
	for( int qn = 0; qn != Q*M; ++qn){
		U1[qn] = &U_data[qn * (D+1)];
	}	
	
	for(int q = 0; q != Q; ++q){
		U[q] = &U1[ q * M];
	}

	/* Z is (Q, N, D+1) matrix */
	double* Z_data = (double*) nu::data(bpn_Z);
	double** Z[Q];
	double* Z1[Q*N];
	
	for( int qn = 0; qn != Q*N; ++qn){
		Z1[qn] = &Z_data[qn * (D+1)];
	}	
	
	for(int q = 0; q != Q; ++q){
		Z[q] = &Z1[ q * N];
	}
	
	/* nz is (Q,N) matrix */
	short* nz_data = (short*) nu::data(bpn_nz);
	short* nz[Q];
	for(int q = 0; q != Q; ++q){
		nz[q] = &nz_data[ q * N];
	}

	hov_ti_reverse(tape_tag, M, N, D, Q, U, Z, nz);

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

// /* from taping.h and taping.c */
// bpn::array wrapped_tapestats(short tape_tag) {
// 	int tape_stats[STAT_SIZE];
// 	tapestats(tape_tag, tape_stats);
// 	return nu::makeNum( tape_stats, STAT_SIZE);
// }


