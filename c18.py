#!/usr/local/Anaconda2023/bin/python3.9


#Q1
# f is representative of f(x,t). The x and t are filled in by x values and t values

#Q2
#step size h is determined by change in t


import my_functions_lib as mfl

# Initial conditions
f = 4
x0 = 2
t0 = 1
t_end = 5
dt = 0.01 ## try two other step sizes

# Solve using Euler method
t_values, x_values = mfl.euler_method(f, x0, t0, t_end, dt)

# Plotting the solution
plt.figure(figsize=(8, 5))
plt.plot(t_values, x_values, label="Euler Approximation", color="b")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Euler Method Solution for dx/dt = x² - x")
plt.grid(True)
plt.legend()
plt.show()		   