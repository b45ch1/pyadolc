//g++ -Iadolc-1.11.0-trunk/include/ -Wl,--rpath -Wl,${PWD}/adolc-1.11.0-trunk/lib -Ladolc-1.11.0-trunk/lib -ladolc -o test -O3 test.cpp

#include "adolc/adolc.h"
#include <iostream>
#include <stdio.h>

using namespace std;


int main(){
    adouble ax1, ax2, ay;
    double *y = myalloc1(1);
    double *x = myalloc1(2);
    x[0] = 3.;
    x[1] = 7.;

    trace_on(11);
    ax1<<=x[0];
    ax2<<=x[1];

    ay = sin(ax1 + ax2*ax1);


    ay>>=*y;
    trace_off();


    double *g = myalloc1(2);
    function(11, 1, 2, x, y );
    gradient(11, 2, x, g );

    tape_doc(11, 1 , 2, x, y );


    printf("function y=[%f]\n", y[0]);
    printf("gradient g=[%f, %f]\n", g[0], g[1]);

    return 0;
}
