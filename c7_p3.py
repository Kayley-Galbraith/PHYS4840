#!/usr/local/Anaconda2023/bin/python

import time
import numpy as np
import sys
import pandas as pd
###################################################
#
# testing np.loadtxt()
#
###################################################
start_numpy = time.perf_counter()
"""
put the action you want to time between the
star and end commands
"""
filename = 'NGC6341.dat'
start = time.perf_counter()
blue, green, red, probability = np.loadtxt(filename, usecols=(8, 14, 26, 32), unpack=True)
print("len(green): ", len(green))
end_numpy  = time.perf_counter()
print('Time to run loadtxt version: ', end_numpy-start_numpy, ' seconds')
###################################################
