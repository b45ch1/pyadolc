//g++ -Iadolc-1.11.0-trunk/include/ -Wl,--rpath -Wl,${PWD}/adolc-1.11.0-trunk/lib -Ladolc-1.11.0-trunk/lib -ladolc -o test -O3 test.cpp

#include "adolc/adolc.h"
#include <iostream>

using namespace std;

int test1(){
	/* this test emulates how pyadolc works with adubs:
	when sin(x) is called in Python, it calls
	adub sin ( const badouble& x ) { }
	which returns not an adub object but just a locint.

	*/

	adouble x(2.);
	cout<<"x.loc="<<x.location<<endl;
	adub y1(sin(x).location);
	cout<<"y1.loc="<<y1.location<<endl;
	cout<<"y1="<<y1<<endl;
	adub y2(5.*x);
	cout<<"y2.loc="<<y2.location<<endl;
	cout<<"y2="<<y2<<endl;

// 	print 'y1.loc=',y1.loc
// 	print 'y1=',y1
// 	y2 = 5.*x
// 	print 'y2.loc=',y2.loc
// 	print 'y1=',y1	
// 	print 'y2=',y2

}


int main(){
// 	adouble a(1.);
//     adub *b = new adub(-a);
// 	b = new adub(operator+(a,2.));
// 	adub *c = new adub(operator+(a,2));
// 	adub *d = new adub(operator+(2.,a));
// 	adub *e = new adub(operator+(2,a));
// 
// 	
// 	cout<<a<<endl;
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

	test1();
	return 0;
} 
