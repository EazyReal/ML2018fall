#Qmonster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#load data
with open('dataset/list.txt') as filelist:
    filename = filelist.readlines()
    filename = filelist[:-2]

#data[time][feature][stack] [200]['a'-'p'][1000]
#data[i].columns = ['a', 'b', ...]


data = []
for i in range(len(filename)):
    data.append(pd.read_csv('dataset/' + filename[i][:-1], names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m","n","o","p"]))

data = [ in data]


K = 20
N = data[0].length
