#!/usr/local/Anaconda2023/bin/python

print(0.1+0.2)
print(repr(0.1+0.2))

from math import sqrt

x = 1.0
y = 1.0 + (1e-14)*sqrt(2)

answer_1 = 1e14*(y-x)
answer_2 = sqrt(2)

print("answer1: ", answer_1)
print("answer2: ", answer_2)
ans = answer_1/answer_2
print((1-ans)*100)
