#include <boost/multi_array.hpp>
#include "ap.hpp"


typedef b::multi_array_ref<double,2> double_matrix_ref;
typedef b::multi_array<double,2> double_matrix;
typedef b::multi_array<double,1> double_vector;
typedef b::multi_array<int,1> int_vector;
typedef b::multi_array<bool,2> bool_matrix;

template<class T>
void print_matrix(T &mat, string msg){
	cout<<msg<<endl;
	int N = mat.shape()[0];
	int M = mat.shape()[1];
	for(int n = 0; n!=N; ++n){
		for(int m = 0; m!=M; ++m){
			cout<<mat[n][m]<<" ";
		}
		cout<<endl;
	}
}

template<class T>
void print_vector(T &vec, string msg){
	cout<<msg<<endl;
	int N = vec.shape()[0];
	for(int n = 0; n!=N; ++n)
		cout<<vec[n]<<" ";
	cout<<endl;
}

bpn::array outer_dot(bpn::array &inMat){
	/* CHECKING INPUT VALUES */
	nu::check_rank(inMat,2);
	vector<intp> shp(nu::shape(inMat));
	int N = shp[0]; // number of data points
	int D = shp[1]; // dimension
	
	/* SETUP VARIABLES */
	double* dataPtr = (double*) nu::data(inMat);
	double_matrix_ref  	x(dataPtr,b::extents[shp[0]][shp[1]]);
	double_matrix		s(b::extents[N][N]);
	
	double tmp;
	for(int n = 0; n!=N; ++n){
		for(int m = 0; m!=N; ++m){
			tmp = 0.;
			for(int d = 0; d!=D; ++d){
				tmp += (x[n][d] - x[m][d]) * (x[n][d] - x[m][d]);
			}
			s[n][m] = tmp;
		}
	}
	
	/* PREPARING RETURN TO PYTHON */
	std::vector<intp> dimensions;
	dimensions.push_back(N);
	dimensions.push_back(N);
	return nu::makeNum( &s[0][0], dimensions);
}

bp::dict ap(bpn::array &inSimilaritiesMatrix, uint maxit, uint convit, double lam){
	
	/* CHECKING INPUT VALUES */
	nu::check_rank(inSimilaritiesMatrix,2);
	vector<intp> shp(nu::shape(inSimilaritiesMatrix));
	int N = shp[0];
	if(N != shp[1]){
		PyErr_SetString(PyExc_ValueError, "Expected a similarity matrix in (N,N) array format!");
		bp::throw_error_already_set();
	}

	/* SETUP VARIABLES */
	double* dataPtr = (double*) nu::data(inSimilaritiesMatrix);
	double_matrix_ref 	s(dataPtr,b::extents[shp[0]][shp[1]]);	//similarities
	double_matrix		r(b::extents[N][N]);			//responsibilities
	double_matrix		a(b::extents[N][N]);			//availabilities
	double_vector		as_max1(b::extents[N]);			//max value of a + s needed in the responsibilities update
	int_vector		as_max1_arg(b::extents[N]);
	double_vector		as_max2(b::extents[N]);			//second max value
	bool_matrix		conv_matrix(b::extents[convit][N]);	//stores for convit successive iterations what data point is an exemplar (1) and which is not (0)
	
	
	double DOUBLE_MIN	= -numeric_limits<double>::max();
	
	/* MAIN ITERATION - AFFINITY PROPAGATION*/
	int it; uint cit;
	for(it= 0; it!=maxit; ++it){
// 		cout<<it<<endl;
		/* UPDATING RESPONSIBILITIES */
		for(int n = 0; n!=N; ++n)as_max1[n] = as_max2[n] = DOUBLE_MIN;
		for(int n_source = 0; n_source!=N; ++n_source){
			for(int n_target = 0; n_target!=N; ++n_target){
				double tmp = a[n_source][n_target] + s[n_source][n_target];
				if(as_max1[n_source]<tmp){
					as_max2[n_source] = as_max1[n_source];
					as_max1_arg[n_source] = n_target;
					as_max1[n_source] = tmp;
				}
				else if(as_max2[n_source]<tmp){
					as_max2[n_source] = tmp;
				}
			}
		}
		for(int n_source = 0; n_source!=N; ++n_source){
			for(int n_target = 0; n_target!=N; ++n_target){
				if(as_max1_arg[n_source]==n_target){
					r[n_source][n_target] = lam*r[n_source][n_target] + (1-lam) *(s[n_source][n_target] - as_max2[n_source]);
				}
				else{
					r[n_source][n_target] = lam*r[n_source][n_target] + (1-lam) *(s[n_source][n_target] - as_max1[n_source]);
				}
			}
		}
// 		print_matrix(r, "R");
		
		/* UPDATING AVAILABILITIES */
		for(int n_target = 0; n_target!=N; ++n_target){
			as_max1[n_target] = 0.;
			for(int n_source = 0; n_source!=N; ++n_source){
				as_max1[n_target] += max(0.,r[n_source][n_target]);
			}
			as_max1[n_target] += min(0., r[n_target][n_target]);
		}
		
		for(int n_source = 0; n_source !=N ; ++n_source){
			for(int n_target = 0; n_target!=N; ++n_target){
				if(n_source==n_target) continue;
				a[n_source][n_target] = lam*a[n_source][n_target] + (1-lam) * (min(0., as_max1[n_target] - max(0., r[n_source][n_target])  ));
			}
			a[n_source][n_source] = lam*a[n_source][n_source] + (1-lam) * ( as_max1[n_source] -  max(0., r[n_source][n_source]) -  min(0., r[n_source][n_source]));
		}
// 		print_matrix(a, "A");
		
		/* CHECKING CONVERGENCE */
		cit = it%convit;
		uint cdiff = 0;
		for(int n = 0; n !=N ; ++n){
			if(a[n][n] + r[n][n]>0.) conv_matrix[cit][n] = true;
			else conv_matrix[cit][n] = false;
		}
		for(uint c = 0; c!=convit; ++c){
			for(int n = 0; n !=N ; ++n){
				if(conv_matrix[c][n] - conv_matrix[cit][n] !=0) cdiff++;
			}
		}
		if(cdiff==0 && it > convit) break;
	}
		
	/* FINDING THE EXEMPLAR IDs */
	vector<int> exids;
	int K;
	for(int n = 0; n !=N ; ++n){
		if(conv_matrix[0][n]==true){
			exids.push_back(n);
		}
	}
	K = exids.size();
		
	/* FINDING CLUSTER LABELS AND THE EXEMPLAR FOR EACH DATA POINT */
	int_vector dpex(b::extents[N]);	 		//index of a data point's exemplar
	int_vector cl(b::extents[N]);			//cluster label in {1,2,...,K}
	for(int n_source = 0; n_source !=N ; ++n_source){
		double tmp=DOUBLE_MIN;
		for(int k = 0; k!= K; ++k){
			if( s[n_source][exids[k]] > tmp){
				tmp 		= s[n_source][exids[k]];
				dpex[n_source]	= exids[k];
				cl[n_source] 	= k;
			}
		}
	}
	for(int k = 0; k != K; ++k){
		dpex[exids[k]] = exids[k];
		cl[exids[k]] = k;
	}

	/* COMPUTING NET SIMILARITY */
	double net_similarity = 0.;
	for(int n_source = 0; n_source !=N ; ++n_source){
		for(int k = 0; k!= K; ++k){
			net_similarity+=s[n_source][exids[k]];
		}
	}
	
	/* COMPUTING AVERAGE PREFERENCE OF THE EXEMPLARS */
	double average_preference = 0.;
	for(int k = 0; k!= K; ++k){
		average_preference += s[dpex[k]][dpex[k]];
	}
	average_preference/=K;
	
	/* COMPUTING NET SELF RESPONSIBILITY OF EXEMPLARS */
	double net_self_responsibility = 0.;
	for(int k = 0; k!= K; ++k){
		net_self_responsibility += r[dpex[k]][dpex[k]];
	}
	
	/* COMPUTING NET RESPONSIBILITY OF EXEMPLARS
	   IE THE DATA POINTS BELONGING TO THE SAME CLUSTER SEND
	   THE RESPONSIBILITY TO THE EXEMPLAR HOW WELL-SUITED IT IS	*/
	double net_responsibility = 0.;
	for(int n_source = 0; n_source !=N ; ++n_source){
		for(int k = 0; k!= K; ++k){
			net_responsibility += r[n_source][dpex[k]];
		}
	}
	
	/* COMPUTING NET AVAILABILITY OF EXEMPLARS
	   IE THE EXEMPLAR SENDS TO THE DATA POINTS HOW WELL-SUITED IT IS*/
	double net_availability = 0.;
	for(int k = 0; k!= K; ++k){
		for(int n_target = 0; n_target !=N ; ++n_target){
			net_availability += a[n_target][dpex[k]];
		}
	}
		
	/* PREPARING OUTPUT TO PYTHON */
	bpn::array ret_dpex 	=  nu::makeNum( &dpex[0], N);
	bpn::array ret_cl	=  nu::makeNum( &cl[0], N);
		
	bp::dict retvals;
	retvals["K"] = K;
	retvals["lam"] = lam;
	retvals["maxit"] = maxit;
	retvals["convit"] = convit;
	retvals["it"] = it;
	retvals["dpex"] = ret_dpex;
	retvals["cl"] = ret_cl;
	retvals["net_similarity"] = net_similarity;
	retvals["average_preference"] = average_preference;
	retvals["net_self_responsibility"] = net_self_responsibility;
	retvals["net_responsibility"] = net_responsibility;
	retvals["net_availability"] = net_availability;
	
	return retvals;
}
