import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Be careful with the file path!
data = loadmat('data/hw4.mat')
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(data['y'])
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def forward_propagate(X, theta1, theta2):
    m = X.shape[0] #data number

    #theta= matrix of out_size * in_size+1 w*X

    #Write codes here
    #np.array([np.dot(np.array(w), X[i]) for i in range(m)])
    #z = a*w.transpose()
    a1 = np.c_[np.ones(m), X] #add an 1 to Xi(first column), type = matrix
    z2 = a1 * theta1.transpose() #m*i * i*o = m*o
    a2 = np.c_[np.ones(m), sigmoid(z2)] 
    z3 = a2 * theta2.transpose()
    h =  sigmoid(z3)
    
    return a1, z2, a2, z3, h
    
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
        
    J = J / m
    J += (float(learning_rate) / (2*m) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2))))
    #no need to regularized thetak[:][0] (bias term)
    
    return J
    
# initial setup
input_size = 400
hidden_size = 10
num_labels = 10
learning_rate = 1
lambda_ = 0.01
# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.2
m = data['X'].shape[0]
X = np.matrix(data['X'])
y = np.matrix(data['y'])
# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))    

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0] #size
   
    #Write codes here
    #J = 0.0
    #grad = grad of each theta(i,j) init= np.zeros(params.shape)

    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    J = cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)

    #calculate delta
    delta3 = h-y #(m, nlab)
    delta2 = np.multiply(delta3*theta2[:,1:], sigmoid_gradient(z2))#(m,n) * (n,hidden+1) (5000*10**10*11)
    ###
    delta3 = np.multiply(delta3, sigmoid_gradient(z3))

    d_b2 = (np.sum(delta3, axis=0)/m).transpose()
    d_b1 = (np.sum(delta2, axis=0)/m).transpose()
    d_w2 = np.dot(delta3.transpose(), z2)/m + lambda_*theta2[:,1:]
    d_w1 = np.dot(delta2.transpose(), X)/m + lambda_*theta1[:,1:]
    
    d_t1 = np.array(np.c_[d_b1, d_w1])
    d_t2 = np.array(np.c_[d_b2, d_w2])
    grad = np.c_[d_t1, d_t2].flatten()

    #for t in range(m):
    #	xt = np.matrix(X[t])

    return J, grad
    
from scipy.optimize import minimize
# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method='TNC', jac=True, options={'maxiter': 250})
      
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))