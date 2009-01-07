//g++ -Iadolc-1.11.0-trunk/include/ -Wl,--rpath -Wl,${PWD}/adolc-1.11.0-trunk/lib -Ladolc-1.11.0-trunk/lib -ladolc -o test -O3 test.cpp

#include "adolc/adolc.h"
#include <iostream>

using namespace std;


int main(){
	adouble a(1.);
    adub *b = new adub(-a);
	b = new adub(operator+(a,2.));
	adub *c = new adub(operator+(a,2));
	adub *d = new adub(operator+(2.,a));
	adub *e = new adub(operator+(2,a));

	
	cout<<a<<endl;
// 	double *y = myalloc1(1);
// 	double x;
// 	x = 1.;
// 	
// 	trace_on(11);
// 	ax<<=x;
// 	ay = ax;
// 	for(int i = 0; i != N; ++i){
// 		ay = ay * ay;
// 	}
// 
// 	ay>>=*y;
// 	trace_off();
// 	
// 	tape_doc(11, 1 , 1, &x, y );
	return 0;
} 
