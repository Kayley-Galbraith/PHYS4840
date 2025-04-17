#!/usr/local/Anaconda2023/bin/python

#####################################
#
# Class 12: Numerical Differentiation II
# Author: M Joyce
#
#####################################

import numpy as np
import matplotlib.pyplot as plt
from math  import tanh, cosh

import sys
sys.path.append('../')
import my_functions_lib as mfl

## compute the instantaneous derivatives
## using the central difference approximation
## over the interval -2 to 2

x_lower_bound = -2.0
x_upper_bound = 2.0

N_samples = 100

#####################
#
# Try different values of h
# What did we "prove" h should be
# for C = 10^(-16) in Python?
#
#######################
h = 0.1 ## what goes here?
h1 = 1 
h2 = 2 

xdata = np.linspace(x_lower_bound, x_upper_bound, N_samples)
xdata1 = np.linspace(x_lower_bound, x_upper_bound, N_samples)
xdata2 = np.linspace(x_lower_bound, x_upper_bound, N_samples)

central_diff_values = []
for x in xdata:
	central_difference = ( mfl.f(x + 0.5*h) - mfl.f(x - 0.5*h) ) / h
	central_diff_values.append(central_difference)

central_diff_values1 = []
for x in xdata1:
	central_difference = ( mfl.f(x + 0.5*h1) - mfl.f(x - 0.5*h1) ) / h1
	central_diff_values.append(central_difference)	

central_diff_values2 = []
for x in xdata2:
	central_difference = ( mfl.f(x + 0.5*h2) - mfl.f(x - 0.5*h2) ) / h2
	central_diff_values.append(central_difference)	

## Add the analytical curve
## let's use the same xdata array we already made for our x values

analytical_values = []
for x in xdata:
	dfdx = mfl.centder(x)
	analytical_values.append(dfdx)


plt.plot(xdata, analytical_values, linestyle='-', color='black')
plt.plot(xdata, central_diff_values, "*", color="green", markersize=8, alpha=0.5)
plt.plot(xdata1, central_diff_values1, "*", color="red", markersize=8, alpha=0.5)
plt.plot(xdata2, central_diff_values2, "*", color="blue", markersize=8, alpha=0.5)
plt.show()
plt.close()