#!/usr/bin/env python
import numpy as npy
import pylab as pyl
import AffinityPropagation as ap

x = pyl.load('data/ToyProblemData.txt')
N = npy.shape(x)[0]


# PREPARING SIMILARITY MATRIX
print 'preparing similarity matrix'
S = ap.outer_dot(x)


#PREPARING PREFERENCES VECTOR
print 'putting preferences in the similarity matrix'
median = npy.median(npy.median(S))
P = npy.repeat(median,N)

#PUTTING PREFERENCES ON THE DIAGONAL OF THE SIMILARITY MATRIX
S[range(N), range(N)] = 70 #P

S = -S

#RUNNING AFFINITY PROPAGATION
print 'running affinity propagation'
dic = ap.ap(S, 100,50,0.5)
print dic.keys()

#print dic['dpex']

#PLOTTING DATASET

