#mwlin
#Kmeans clustering with L2 distance on stock history data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import cluster
import pickle

#hyper parms and numbers
K = 20
N = 0

#load data
with open('dataset/list.txt') as filelist:
    filename = filelist.readlines()
    filename = filename[:-2] #sed no work

#data[time][feature][stack] [200][''][1000]
#data[i].columns = ['id', 'b', ...]
data = []
for i in range(len(filename)):
    data.append(pd.read_csv('dataset/' + filename[i][:-1], names = ["id", "b", "c", "d", "e", "f", "g", "h", "price", "sign", "dprice", "l", "m","n","o","p"]))
for i in range(len(data)):
    data[i] = data[i].replace('--', '0.0')
    
#generate stock list, by transaction count > 500 on any day
def getstocklist(filename):
    name = []
    flag = 0
    #filename = 'dataset/20170410.csv'  
    try:
        csvfile = open(filename, 'r')
    except:
        return []
    for row in csv.reader(csvfile, delimiter=','): 
        if "0050" in row[0]:
            flag = 1
        if flag and int(row[3].replace(',',''))>500:
            name.append(row[0])
        if "9958" in row[0]:
            flag = 0
    return name

stocklist = set([])
for i in range(0, len(filename)):
    stocklist = stocklist.union(set(getstocklist('dataset/' + filename[i][:-1])))
stocklist = list(stocklist)

N = len(stocklist)


#data for kmeans clustering
#cdata[N][days]
cdata = []
for stocki in range(N):
    svec = []
    sname = stocklist[stocki] #the name
    for dayi in range(len(data)):
        ddata = data[dayi]
        if len(ddata[ddata['id'] == sname]):
            #change rate
            try:
                ret = float(ddata[ddata['id'] == sname]['dprice']) / float(ddata[ddata['id'] == sname]['price'])
            except:
                ret = 0.0
            if(np.array(ddata[ddata['id'] == sname]['sign'])[0] ==  '-'):
                ret = ret * -1.0
            svec.append(ret)
        else:
            svec.append(0.0)
    cdata.append(svec)


#kmeans with sklearn
kmeans_fit = cluster.KMeans(n_clusters = K).fit(cdata)

cluster_labels = kmeans_fit.labels_

print(cluster_labels)

#turn to res[k] = stack ids of the cluster
res = []

for i in range(K):
    res.append([])
for x in range(N):
    res[cluster_labels[x]].append(stocklist[x])

for i in range(K):
    print(res[i])



with open('cdata_all.pkl', 'wb+') as fo:
    pickle.dump(cdata, fo)
with open('cluster_labels_all.pkl', 'wb+') as fo:
    pickle.dump(cluster_labels, fo)
with open('res_all.pkl', 'wb+') as fo:
    pickle.dump(res, fo)

