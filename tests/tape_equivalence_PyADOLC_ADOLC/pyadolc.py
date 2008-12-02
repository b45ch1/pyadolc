#!/usr/bin/env python
from adolc import *
import numpy as npy

N = 20

ay = adouble(0.)
ax = adouble(0.)
x = 1.


trace_on(10)
ax.is_independent(x)
ay = ax
for i in range(N):
	ay = ay * ay
	#if i%1000 == 0:
		#tmpay <<= ay
		#del(ay)
		#ay = tmpay
		
depends_on(ay)
trace_off()
tape_to_latex(10,npy.array([x]),npy.array([0]))