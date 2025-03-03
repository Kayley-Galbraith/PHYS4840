/mnt/c/Users/kayle/PHYS4840/python
#Homework 3
# PHYS4840
#Kayley Galbraith


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from math import factorial

#Problem 0


# Example usage with array data
def trapezoidal(y_values, x_values, N):
    """
    Approximates the integral using trapezoidal rule for given y_values at given x_values.

    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        N (int): Number of intervals.

    Returns:
        float: The approximated integral.
    """
    a, b = x_values[0], x_values[-1]
    h = (b - a) / N

    integral = (1/2) * (y_values[0] + y_values[-1])  # First and last terms

    for k in range(1, N):
        xk = a + k * h  # Compute x_k explicitly
        yk = np.interp(xk, x_values, y_values)  # Interpolate y at x_k manually in loop
        integral += yk

    integral *= h
    return integral


# Simpson's rule for array data
def simpsons(y_values, x_values, N):
    """
    Approximates the integral using Simpson's rule for given y_values at given x_values.

    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        N (int): Number of intervals (must be even).

    Returns:
        float: The approximated integral.
    """
    if N % 2 != 0:
        raise ValueError("N must be even for Simpson's rule.")

    a, b = x_values[0], x_values[-1]
    h = (b - a) / N

    integral = y_values[0] + y_values[-1]  # First and last terms

    for k in range(1, N, 2):  # Odd indices (weight 4)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 4 * yk

    for k in range(2, N, 2):  # Even indices (weight 2)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 2 * yk

    return (h / 3) * integral  # Final scaling


# Romberg integration for array data
def romberg(y_values, x_values, max_order):
    """
    Approximates the integral using Romberg's method for given y_values at given x_values.

    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        max_order (int): Maximum order (controls accuracy).

    Returns:
        float: The approximated integral.
    """
    R = np.zeros((max_order, max_order))
    a, b = x_values[0], x_values[-1]

    # First trapezoidal estimate
    R[0, 0] = (b - a) * (y_values[0] + y_values[-1]) / 2.0

    for i in range(1, max_order):
        N = 2**i
        h = (b - a) / N
        
        sum_new_points = sum(np.interp(a + (k - 0.5) * h, x_values, y_values) for k in range(1, N + 1))
        R[i, 0] = 0.5 * R[i - 1, 0] + (b - a) * sum_new_points / N

        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)

    return R[max_order - 1, max_order - 1]


def timing_function(integration_method, x_values, y_values, integral_arg):
    """
    Times the execution of an integration method.

    Parameters:
        integration_method (function): The numerical integration function.
        x_values (array-like): The x values.
        y_values (array-like): The corresponding y values.
        integral_arg (int, optional): EITHER Number of intervals to use (Simpson/Trapz) OR the maximum order of extrapolation (Romberg).

    Returns:
        tuple: (execution_time, integration_result)
    """
    start_time = time.perf_counter()
    result = integration_method(y_values, x_values, integral_arg)
    end_time = time.perf_counter()
    return end_time - start_time, result


# Function to integrate
def function(x):
    return x * np.exp(-x)

# Precompute data for fair comparisons
x_data = np.linspace(0, 1, 1000)  # High-resolution x values
y_data = function(x_data)

# Testing parameters
N = 100  # Number of intervals
max_order = 10  # Romberg's accuracy level

# Measure timing for custom methods
trap_time, trap_result = timing_function(trapezoidal, x_data, y_data, N)
simp_time, simp_result = timing_function(simpsons, x_data, y_data, N)
romb_time, romb_result = timing_function(romberg, x_data, y_data, max_order)

# True integral value
true_value = 0.26424111765711535680895245967707826510837773793646433098432639660507700851

# Compute errors
trap_error = (abs(true_value - trap_result) / true_value) * 100
simp_error = (abs(true_value - simp_result) / true_value) * 100
romb_error = (abs(true_value - romb_result) / true_value) * 100

# Print results with error analysis
print("\nIntegration Method Comparison")
print("=" * 80)  # why 80? https://peps.python.org/pep-0008/
print(f"{'Method':<25}{'Result':<20}{'Error (%)':<20}{'Time (sec)':<15}")
print("-" * 80)
print(f"{'Custom Trapezoidal':<25}{trap_result:<20.8f}{trap_error:<20.8e}{trap_time:<15.6f}")
print(f"{'Custom Simpsons':<25}{simp_result:<20.8f}{simp_error:<20.8e}{simp_time:<15.6f}")
print(f"{'Custom Romberg':<25}{romb_result:<20.8f}{romb_error:<20.8e}{romb_time:<15.6f}")
print("=" * 80)


#Problem 0b

# Trapezoidal parameters
Ns_trap = [2**i for i in range(1, 11)]  # 2, 4, 8, ..., 1024
errors_trap, times_trap = [], []
for N in Ns_trap:
    time_taken, result = timing_function(trapezoidal, x_data, y_data, N)
    error = abs(true_value - result) / true_value * 100
    errors_trap.append(error)
    times_trap.append(time_taken)

# Simpsons parameters
Ns_simp = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
errors_simp, times_simp = [], []
for N in Ns_simp:
    time_taken, result = timing_function(simpsons, x_data, y_data, N)
    error = abs(true_value - result) / true_value * 100
    errors_simp.append(error)
    times_simp.append(time_taken)

# Romberg parameters
orders_romb = list(range(1, 11))
errors_romb, times_romb = [], []
for order in orders_romb:
    time_taken, result = timing_function(romberg, x_data, y_data, order)
    error = abs(true_value - result) / true_value * 100
    errors_romb.append(error)
    times_romb.append(time_taken)

# Trapezoidal
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.semilogx(Ns_trap, errors_trap, 'b-o')
plt.xlabel('Number of Intervals (N)')
plt.ylabel('Error (%)')
plt.title('Trapezoidal Method: Error vs Number of Intervals')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.loglog(times_trap, errors_trap, 'r-o')
plt.xlabel('Time (seconds)')
plt.ylabel('Error (%)')
plt.title('Trapezoidal Method: Error vs Compute Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# Simpson's
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(Ns_simp, errors_simp, 'b-o')
plt.xlabel('Number of Intervals (N)')
plt.ylabel('Error (%)')
plt.title('Simpson’s Method: Error vs Number of Intervals')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.loglog(times_simp, errors_simp, 'r-o')
plt.xlabel('Time (seconds)')
plt.ylabel('Error (%)')
plt.title('Simpson’s Method: Error vs Compute Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# Romberg
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(orders_romb, errors_romb, 'b-o')
plt.xlabel('Romberg Order')
plt.ylabel('Error (%)')
plt.title('Romberg Method: Error vs Order')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.loglog(times_romb, errors_romb, 'r-o')
plt.xlabel('Time (seconds)')
plt.ylabel('Error (%)')
plt.title('Romberg Method: Error vs Compute Time')
plt.grid(True)
plt.tight_layout()
plt.show()


#problem 1
gaia_data = pd.read_csv('GAIA.csv')
vega_data = pd.read_csv('Vega_SED.csv')

# x and y values
x_gaia = gaia_data['x'].values
y_gaia = gaia_data['y'].values

x_vega = vega_data['x'].values
y_vega = vega_data['y'].values

# Define a function to compute the area and time for each method
def compute_area(x_values, y_values, intervals, max_order):
    areas = {}
    times = {}

    # Trapezoidal
    trap_time, trap_area = timing_function(trapezoidal, x_values, y_values, intervals)
    areas['Trapezoidal'] = trap_area
    times['Trapezoidal'] = trap_time

    # Simpson's
    try:
        simp_time, simp_area = timing_function(simpsons, x_values, y_values, intervals)
        areas['Simpson'] = simp_area
        times['Simpson'] = simp_time
    except ValueError as e:
        areas['Simpson'] = str(e)
        times['Simpson'] = None

    # Romberg
    romb_time, romb_area = timing_function(romberg, x_values, y_values, max_order)
    areas['Romberg'] = romb_area
    times['Romberg'] = romb_time

    return areas, times

# Number of intervals and maximum order for Romberg
N = 100  # Number of intervals
max_order = 10  # Romberg's accuracy level

# Compute area for GAIA
areas_gaia, times_gaia = compute_area(x_gaia, y_gaia, N, max_order)

# Compute area for Vega
areas_vega, times_vega = compute_area(x_vega, y_vega, N, max_order)

print(f"Trapezoidal Area: {areas_gaia['Trapezoidal']}")
print(f"Simpson Area: {areas_gaia['Simpson']}")
print(f"Romberg Area: {areas_gaia['Romberg']}")

print(f"Trapezoidal Area: {areas_vega['Trapezoidal']}")
print(f"Simpson Area: {areas_vega['Simpson']}")
print(f"Romberg Area: {areas_vega['Romberg']}")

# GAIA
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(x_gaia, y_gaia, label='GAIA')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('GAIA Data Curve')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_vega, y_vega, label='Vega SED', color='orange')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Vega SED Data Curve')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


#Part 1b
#Simpson's rule requires an even number of intervals
#Romberg Integration requires a large amount of data points
#You would need to load the data, apply the integration method, Simpson's rule requires a try-except for odd N's and Romberg requires a max_order to get the appropriate amount of data.

#Problem 2
#Numerical Problem: \int^3_0 {\sqrt{r+x^3}}
def function(x):
    return np.sqrt(4 + x**3)

def simpsons_rule(f, a, b, N):
    if N % 2 != 0:
        raise ValueError("N must be even for Simpson's rule.")
    
    h = (b - a) / N
    integral = f(a) + f(b)
    
    for i in range(1, N, 2):
        integral += 4 * f(a + i * h)
    
    for i in range(2, N, 2):
        integral += 2 * f(a + i * h)
    
    return (h / 3) * integral
a = 0
b = 3
N = 100  # Number of intervals (must be even)

result = simpsons_rule(function, a, b, N)
print(f"The approximate integral is: {result}")

#Simpson's rule has better accuracy than trapezoidal rule, but has a better efficiency than the Romberg integration

#Problem 3

def f(x):
    return x**2

x_values = np.array([1, 2, 3, 4, 5])  # You can change these values as needed
n = len(x_values)

s = np.sum([f(x_i) for x_i in x_values])
print(f"Sum of f(x) where f(x) = x^2: {s}")

x_mean = np.mean(x_values)
print(f"Mean of the set S: {x_mean}")

n_factorial = factorial(n)
print(f"Factorial of n ({n}): {n_factorial}")

#Problem 4
#part a

def intg(x):
    if abs(x) < 1e-12:
        return 1.0
    else:
        return (np.sin(x) ** 2) / (x ** 2)

def ad_int(a, b, epsilon):
    delta = epsilon / (b - a)
    
    def step(x1, x2, f1, f2):
        h = x2 - x1
        xm = 0.5 * (x1 + x2)
        fm = intg(xm)
        I1 = 0.5 * h * (f1 + f2)
        I2 = 0.25 * h * (f1 + 2 * fm + f2)
        error = abs(I2 - I1) / 3
        target_error = h * delta
        
        if error <= target_error:
            return (h / 6) * (f1 + 4 * fm + f2)
        else:
            return step(x1, xm, f1, fm) + step(xm, x2, fm, f2)
    
    f1 = intg(a)
    f2 = intg(b)
    return step(a, b, f1, f2)

# Parameters
a = 0.0
b = 10.0
epsilon = 1e-4

result = ad_int(a, b, epsilon)
print(f"Integral result: {result:.6f}")
print(f"Target accuracy: {epsilon}")

#Part c
def ad_int_plot(a, b, epsilon):
    delta = epsilon / (b - a)
    intervals = []
    
    def step(x1, x2, f1, f2):
        h = x2 - x1
        xm = 0.5 * (x1 + x2)
        fm = intg(xm)
        I1 = 0.5 * h * (f1 + f2)
        I2 = 0.25 * h * (f1 + 2 * fm + f2)
        error = abs(I2 - I1) / 3
        target_error = h * delta
        
        if error <= target_error:
            intervals.append((x1, x2))
            return (h / 6) * (f1 + 4 * fm + f2)
        else:
            return step(x1, xm, f1, fm) + step(xm, x2, fm, f2)
    
    f1 = integrand(a)
    f2 = integrand(b)
    result = step(a, b, f1, f2)
    
    # Generate plot
    x_vals = np.linspace(a, b, 1000)
    y_vals = [integrand(x) for x in x_vals]
    plt.plot(x_vals, y_vals, label='Integrand')
    
    # Extract slice points
    slice_points = set()
    for interval in intervals:
        slice_points.add(interval[0])
        slice_points.add(interval[1])
    slice_points = sorted(slice_points)
    plt.plot(slice_points, np.zeros_like(slice_points), 'ro', markersize=2, label='Slice endpoints')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Adaptive Integration Slices')
    plt.legend()
    plt.show()
    
    return result

result = ad_int_plot(a, b, epsilon)
print(f"Integral result: {result:.6f}")

#part b
#It is used to save computing time for the function. By not having to calculate 
# the values, it can run more efficiently.
