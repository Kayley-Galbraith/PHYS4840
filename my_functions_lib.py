#!/usr/local/Anaconda2023/bin/python3.9

import matplotlib.pyplot as plt
import numpy as np
def my_function(vector):
	a = vector[0]
	b = vector[1]
	c = vector[2]

	return np.linalg.norm(vector)

def y(x):
	y = 2.0*x**3.0
	return y