#!/usr/local/Anaconda2023/bin/python

import numpy as np
import matplotlib.pyplot as plt 
import sys
import pandas as pd 

blue, green, red = [], [], []

filename = 'NGC6341.dat'
with open(filename, 'r') as file:
	for line in file:
		if line.startswith('#'):
			continue

		columns = line.split()
		
		blue.append(float(columns[8]))
		green.append(float(columns[14]))
		red.append(float(columns[26]))

blue = np.array(blue)
green = np.array(green)
red = np.array(red)			

print(blue,green,red)

plt.plot(red, blue, marker = ".")
plt.show()

quality_cut_red = np.where(red > 5)