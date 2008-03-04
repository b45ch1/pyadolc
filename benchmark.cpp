#include <vector>
#include <sys/time.h>
#include <boost/multi_array.hpp>

#include "adolc/adolc.h"
#include "numerical_helper_functions.hpp"

using namespace std;
using namespace boost;
typedef boost::multi_array<double,2> double_matrix;

/* return time in seconds */
struct timeval tv;
double mtime(void)
{
gettimeofday(&tv,NULL);
return (double)(tv.tv_sec + (tv.tv_usec / 1000000.));
}

ostream& operator<<(ostream& os, const double_matrix& A){
	for(int n = 0; n != (int) A.shape()[0]; ++n){
		for(int m = 0; m != (int) A.shape()[1]; ++m){
			os<<A[n][m]<<"\t";
		}
		os<<endl;
	}
	return os;
}

ostream& operator<<(ostream& os, const vector<double> x){
	for(int n = 0; n!= (int)x.size(); ++n){
		os<<x[n]<<"\t";
	}
	os<<endl;
	return os;
}

/* function f: \mathbb R^n \longrightarrow \mathbb R */
void double_speelpenning(const vector<double> &x, vector<double> &y){
	double tmp(1);
	for(int i = 0; i != x.size(); ++i){
		tmp*=x[i];
	}
	y[0] = tmp;
}
void adouble_speelpenning(const vector<adouble> &x, vector<adouble> &y){
	adouble tmp(1);
	for(int i = 0; i != x.size(); ++i){
		tmp*=x[i];
	}
	y[0] = tmp;
}

void double_matrix_vector_multiplication(const vector<double> &x, vector<double> &y){
	int N = x.size();
	int M = y.size();
	double_matrix A(extents[M][N]);
	for(int m = 0; m != M; ++m){
		for(int n = 0; n != N; ++n){
			A[m][n] = 1./N + (n==m);
		}
	}

	for(int m = 0; m != M; ++m){
		y[m] = 0.;
	}
	for(int m = 0; m != M; ++m){
		for(int n = 0; n != N; ++n){
			y[m]+= A[m][n]*x[n];
		}
	}
}
void adouble_matrix_vector_multiplication(const vector<adouble> &x, vector<adouble> &y){
	int N = x.size();
	int M = y.size();
	double_matrix A(extents[M][N]);
	for(int m = 0; m != M; ++m){
		for(int n = 0; n != N; ++n){
			A[m][n] = 1./N + (n==m);
		}
	}

	for(int m = 0; m != M; ++m){
		y[m] = 0.;
	}
	for(int m = 0; m != M; ++m){
		for(int n = 0; n != N; ++n){
			y[m]+= A[m][n]*x[n];
		}
	}
}


void benchmark(void (*adouble_f)(const vector<adouble>&, vector<adouble> &), void (*double_f)(const vector<double>&, vector<double> &),int N, int M, string message){
	double start_time;
	double runtime_taping;
	double runtime_adolc_function;
	double runtime_function;
	double runtime_gradient;
	double runtime_jacobian;

	vector<double> x(N);
	vector<double> g(N);
	vector<adouble> ax(N); 		/* active variables */
	vector<adouble> ay(M);
	vector<double> y(M);			/* y = f(x) */
	vector<double> y_normal(M);
	vector<double> directional_derivative(M);
	double **J = myalloc2(M,N);


	printf("%s", message.c_str());


	for(int n = 0; n!=N; ++n){
		 x[n] = 1./(n+1);
	}
	start_time = mtime();
	trace_on(1);
		for(int n = 0; n!=N; ++n) ax[n].operator<<=(x[n]);
		adouble_f(ax, ay);
		for(int m = 0; m!=M; ++m) ay[m]>>= y[m];
	trace_off();
	runtime_taping = mtime() - start_time;
	printf("Adolc\tfunction taping:\t........\telapsed time: %f\n",runtime_taping);


 	start_time = mtime();
	function(1, M, N, &x[0], &y[0]);
	runtime_adolc_function = mtime() - start_time;
	printf("Adolc\tfunction evaluation:\t%f\telapsed time: %f\n", y[0], runtime_adolc_function);


	start_time = mtime();
	double_f(x,y_normal);
	runtime_function = mtime() - start_time;
	printf("normal\tfunction evaluation:\t%f\telapsed time: %f\n",y_normal[0],runtime_function);


	start_time = mtime();
	jacobian(1, M, N, &x[0], J);
	runtime_jacobian = mtime() - start_time;
	printf("jacobian evaluation:\t\t........\telapsed time: %f\n", runtime_jacobian);

	if(M==1){
		start_time = mtime();
		gradient(1,N,x.data(),g.data());
		runtime_gradient = mtime() - start_time;
		printf("gradient evaluation:\t\t........\telapsed time: %f\n", runtime_gradient);
	}
	
}




int main(int argc, char* argv[]){
	int N;
	int M;

	if(argc == 3){
		N = atoi(argv[1]);
		M = atoi(argv[2]);
	}
	else{
		N = 10;
		M = 1;
	}
	cout<<"N="<<N<<" M="<<M<<endl;
	benchmark(adouble_speelpenning, double_speelpenning,N,1, "\n\nspeelpenning:\n");
	benchmark(adouble_matrix_vector_multiplication, double_matrix_vector_multiplication,N,M, "\n\nmatrix vector multiplication:\n");
	
	/* tape_doc(tag, m, n, x[n], y[m]) */   
// 	tape_doc(1, M, N, x, y);
	
	return EXIT_SUCCESS;
}
