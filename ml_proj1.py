from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
import graphviz
import pydotplus
iris = datasets.load_iris()
labels = [ 'setosa' , 'versicolor' , 'virginica' ]
sepal_length = iris.data[:,0]
sepal_width = iris.data[:,1]
petal_length = iris.data[:,2]
petal_width = iris.data[:,3]
df_train, df_test= train_test_split(iris.data, test_size=0.25, random_state=0)
dt_train, dt_test= train_test_split(iris.target, test_size=0.25, random_state=0)

#DecisionTree with sepal_length
clf1 = tree.DecisionTreeClassifier()
X1 = np.array(([df_train[:,0]])).reshape(-1, 1)
clf1 = clf1.fit(X1, dt_train)
#DecisionTree with sepal_width
clf2 = tree.DecisionTreeClassifier()
X2 = np.array(([df_train[:,1]])).reshape(-1, 1)
clf2 = clf2.fit(X2, dt_train)
#DecisionTree with petal_length
clf3 = tree.DecisionTreeClassifier()
X3 = np.array(([df_train[:,2]])).reshape(-1, 1)
clf3 = clf3.fit(X3, dt_train)
#DecisionTree with petal_width
clf4 = tree.DecisionTreeClassifier()
X4 = np.array(([df_train[:,3]])).reshape(-1, 1)
clf4 = clf4.fit(X4, dt_train)

X1_test = np.array([df_test[:,0]]).reshape(-1, 1)
X2_test = np.array([df_test[:,1]]).reshape(-1, 1)
X3_test = np.array([df_test[:,2]]).reshape(-1, 1)
X4_test = np.array([df_test[:,3]]).reshape(-1, 1)


#performance its self
randomTree0 = (clf1.predict_proba(X1) + clf2.predict_proba(X2) + clf3.predict_proba(X3) + clf4.predict_proba(X4)) / 4
ans0 = np.empty(len(randomTree0), dtype=int)
for index in range(len(randomTree0)):
    if randomTree0[index,0] >= randomTree0.max(axis=1)[index] :
        ans0[index] = '0'
    elif randomTree0[index,1] >= randomTree0.max(axis=1)[index] :
        ans0[index] = '1'
    else :
        ans0[index] = '2'
print('Self Performance is ',metrics.accuracy_score(dt_train, ans0))

#Accuracy 
randomTree = (clf1.predict_proba(X1_test) + clf2.predict_proba(X2_test) + clf3.predict_proba(X3_test) + clf4.predict_proba(X4_test)) / 4
ans = np.empty(len(randomTree), dtype=int)
for index in range(len(randomTree)):
    if randomTree[index,0] >= randomTree.max(axis=1)[index] :
        ans[index] = '0'
    elif randomTree[index,1] >= randomTree.max(axis=1)[index] :
        ans[index] = '1'
    else :
        ans[index] = '2'
print('Accuracy is ',metrics.accuracy_score(dt_test, ans))

#confusion matrix
array = confusion_matrix(dt_test, ans)
df_cm = pd.DataFrame(array, index = labels,
                  columns = labels)
df_cm.columns.name = 'Predicted'
df_cm.index.name = 'Actual'
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
