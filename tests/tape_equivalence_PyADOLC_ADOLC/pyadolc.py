#!/usr/bin/env python
from adolc import *
import numpy as npy
N = 20

ay = adouble(0.)
ax = adouble(0.)
x = 0.99

# usual way: leads to increasing locints
trace_on(9)
independent(ax)
ay = ax
for i in range(N):
    ay = ay * ay

dependent(ay)
trace_off()


# if necessary: call the garbage collector and therefore start with smaller locints
del(ax)
del(ay)

trace_on(10)
ax = adouble(1)
independent(ax)
ay = ax
for i in range(N):
    ay <<= ay * ay


dependent(ay)
trace_off()

tape_to_latex(9,npy.array([x]),npy.array([0]))
tape_to_latex(10,npy.array([x]),npy.array([0]))