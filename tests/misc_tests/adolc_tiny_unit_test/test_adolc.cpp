//g++ -Iadolc-1.11.0-trunk/include/ -Wl,--rpath -Wl,${PWD}/adolc-1.11.0-trunk/lib -Ladolc-1.11.0-trunk/lib -ladolc -o test -O3 test.cpp

#include "adolc.h"
#include <iostream>
#include <string>

using namespace std;


int test_condassign_on_adouble(){
	adouble a(2.);
	adouble cond(3.);
	adouble cond2(-3.);
	adouble b(4.);
	adouble c(5.);
	
	condassign(a,cond,b,c);
	if(a != b)return -1;
	
	condassign(a,cond2,b,c);
	if(a != c)return -1;
	
	return 0;
}

int test_condeqassign_on_adouble(){
	adouble a(2.);
	adouble cond(0.);
	adouble cond2(-3.);
	adouble b(4.);
	adouble c(5.);

	condeqassign(a,cond,b,c);
	if(a != b)return -1;

	condeqassign(a,cond2,b,c);
	if(a != c)return -1;

	return 0;
}

int test_condassign_on_double(){
	double a(2.);
	double cond(3.);
	double cond2(-3.);
	double b(4.);
	double c(5.);
	
	condassign(a,cond,b,c);
	if(a != b) return -1;
	
	condassign(a,cond2,b,c);
	if(a != c) return -1;
	
	return 0;
}

int test_condeqassign_on_double(){
	double a(2.);
	double cond(0.);
	double cond2(-3.);
	double b(4.);
	double c(5.);

	condeqassign(a,cond,b,c);
	if(a != b) return -1;

	condeqassign(a,cond2,b,c);
	if(a != c) return -1;

	return 0;
}

void check(int error, string test_name){
	if(error != 0){
		cout<<"Test "<<test_name<<" : failed!"<<endl;
	}
	else{
		cout<<"Test "<<test_name<<" : OK!"<<endl;
	}
}

int main(){
	check(test_condassign_on_adouble(),"test_condassign_on_adouble");
	check(test_condeqassign_on_adouble(),"test_condeqassign_on_adouble");
	check(test_condassign_on_double(),"test_condassign_on_double");
	check(test_condeqassign_on_double(),"test_condeqassign_on_double");
	return 0;
}
