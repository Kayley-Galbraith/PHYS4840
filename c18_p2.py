#!/usr/local/Anaconda2023/bin/python3.9


from math import sin
from numpy import arange
import matplotlib.pyplot as plt
from pylab import plot,xlabel, ylabel, show

def f(x,t):
	return -x**3 + sin(t)

a = 0.0
b = 10.0
N = 10
N2 = 20
N3= 50
N4 = 100
h = (b-a)/N 
h1 = 0.5 * h
h2 = 0.2 * h 
h3 = 0.1 * h

tpoints = arange(a,b,h)
xpoints = []

tpoints1 = arange(a,b,h1)
xpoints1 = []

tpoints2 = arange(a,b,h2)
xpoints2 = []

tpoints3 = arange(a,b,h3)
xpoints3 = []
x = 0.0
for t in tpoints:
	xpoints.append(x)
	k1 = h*f(x,t)
	k2 = h*f(x+0.5*k1, t+0.5*h)
	x += k2

x = 0.0
for t in tpoints1:
	xpoints1.append(x)
	k1 = h1*f(x,t)
	k2 = h1*f(x+0.5*k1, t+0.5*h1)
	x += k2	

x = 0.0
for t in tpoints2:
	xpoints2.append(x)
	k1 = h2*f(x,t)
	k2 = h2*f(x+0.5*k1, t+0.5*h2)
	x += k2	
x = 0.0
for t in tpoints3:
	xpoints3.append(x)
	k1 = h3*f(x,t)
	k2 = h3*f(x+0.5*k1, t+0.5*h3)
	x += k2

plt.plot(tpoints,xpoints, label = "N=10")
plot(tpoints1,xpoints1)
plot(tpoints2,xpoints2)
plot(tpoints3,xpoints3)
plot.legend()
xlabel("t")
ylabel("x(t)")
show()		