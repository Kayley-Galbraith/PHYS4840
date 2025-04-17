#!/usr/local/Anaconda2023/bin/python3.9

#####################################
#
# Class 14: Matrices and Linear algebra 
# Author: M Joyce
#
#####################################

import numpy as np
import my_functions_lib as mfl 


A = np.array([ [2, -1, 3,],\
			   [-1, 4, 5], 
			   [3,  5, 6] ],float)

print(mfl.qr_decomposition(A))



#Q is orthogonal is q^T Q = I






'''

by importing and using the QR decomposition 
algorithm in my_functions_lib.py:
1) Find Q and R
2) Confirm that Q is orthogonal
3) Confirm that R is upper triangular
4) Confirm that the matrix A introduced in eigenvalues.py
can indeed be reconstructed by the dot product 
of matrices Q and R
'''