#!/usr/bin/env python
from adolc import *
import numpy


def test_tape_to_latex():
	def f(avec):
		a = avec[0]*avec[-1]
		c =avec[0]*avec[0]
		if isinstance(a,badouble):
			print avec[0].loc
			print avec[1].loc
			print a.loc
			print c.loc
		return a


	N = 2
	x = numpy.array([1.,2.])
	ax = numpy.array([adouble(0.),adouble(0.)])

	trace_on(1)
	for n in range(N):
		ax[n].is_independent(x[n])
	ay = f(ax)
	depends_on(ay)
	trace_off()

	tape_to_latex(1,x,numpy.array([0]))

def test_tape_stats():
	def f(avec):
		a = avec[0]*avec[-1]
		c =avec[0]*avec[0]
		if isinstance(a,badouble):
			print avec[0].loc
			print avec[1].loc
			print a.loc
			print c.loc
		return a
	
	N = 2
	x = numpy.array([1.,2.])
	ax = numpy.array([adouble(0.),adouble(0.)])

	trace_on(1)
	for n in range(N):
		ax[n].is_independent(x[n])
	ay = f(ax)
	depends_on(ay)
	trace_off()

	print tapestats(1)

if __name__ == "__main__":
	test_tape_stats()