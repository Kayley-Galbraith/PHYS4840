#!/usr/local/Anaconda2023/bin/python3.9

#find x = 2-e^(-x)
import numpy as np
from math import exp
from math import sqrt
from math import log
x = 0.5
for i in range(10):
	x = 2-exp(-x)
	print(x)
A=0.5
B = 0.5	
A = exp(1-A**2)
B = sqrt(1-log(B))

print("")
print('The A values are:')
for i in range(10):
	print(A)
print('')	
print('The B values are:')	
for i in range(10):
	print(B)