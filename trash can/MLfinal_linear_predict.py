import sys 
import os
import pandas as pd

from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt

from sklearn.linear_model import LinearRegression

#data = ? data[stack][price]
#N = stack num
#k = fearture
#lm[] = lm for stacks

for stack_i in data:
	stack = data[stack_i]
	lm[stack_i] = LinearRegression()
	for i in range (N - k):
		lm.fit(stack[i], )




