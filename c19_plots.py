#!/usr/local/Anaconda2023/bin/python3.9

import numpy as np 
import matplotlib.pyplot as plt
import sys

sys.path.append('rk2_results1.dat')

filename = 'rk2_results1.dat'

x_values,t_values = np.loadtxt(filename, usecols=(t,x), unpack=True)

plt.plot(x_values,t_values)