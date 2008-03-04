//g++ -Iadolc-1.11.0-trunk/include/ -Wl,--rpath -Wl,${PWD}/adolc-1.11.0-trunk/lib -Ladolc-1.11.0-trunk/lib -ladolc -o test -O3 test.cpp

#include "adolc/adolc.h"
#include <iostream>

using namespace std;


int main(){
	int N = 10000000;
	adouble ax,ay;
	double *y = myalloc1(1);
	double x;
	x = 1.;
	
	trace_on(11);
	ax<<=x;
	ay = ax;
	for(int i = 0; i != N; ++i){
		ay = ay * ay;
	}

	ay>>=*y;
	trace_off();
	
// / 	tape_doc(11, 1 , N, x, y );	
	return 0;
} 
