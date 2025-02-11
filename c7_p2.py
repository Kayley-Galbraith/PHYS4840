#!/usr/local/Anaconda2023/bin/python

import timeit
import numpy as np
setup_code = "import numpy as np; my_array = np.arange(100000)"
a = timeit.timeit("sum([x**2 for x in range(100000)])", setup = setup_code, number = 100)
b = timeit.timeit("np.sum(my_array**2", setup+setup_code, number = 100)
c = timeit.timeit("sum([x**2 for x in range(100000)])", setup = setup_code, number = 1000)
d = timeit.timeit("np.sum(my_array**2", setup+setup_code, number = 1000)

print(a)
print(b)
print(c)
print(d)
