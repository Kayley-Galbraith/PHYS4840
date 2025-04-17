#!/usr/local/Anaconda2023/bin/python3.9

import numpy as np

def f(x):
    return x**2  # Some function thats easy to integrate by hand and hence verify

# Number of points (n) for Gauss-Legendre Quadrature
n = 1000

# Compute the Gauss-Legendre Quadrature points (roots of the Legendre polynomial) and weights
root, weights = np.polynomial.legendre.leggauss(n)

#print the roots and weights for the points
print('root', root)
print('weights', weights)

# Compute the integral approximation manually using a for loop
#iterating through each legendre polynomial
integral_approx = 0
for i in range(n):
    point = root[i]
    weight = weights[i]
    function_value = f(point)
    weighted_value = weight * function_value
    integral_approx = integral_approx + weighted_value

    #grab the root for this polynomial
    #grap the weight for this polynomial
    #Evaluate function at the root
    # Apply weight
    # append to running sum

exact_integral = 2/3


# Print final comparison
print("\nFinal Results:")
print(f"Approximated integral using Gauss-Legendre Quadrature: {integral_approx}")
print(f"Exact integral: {exact_integral}")
print(f"Error: {abs(integral_approx - exact_integral)}")