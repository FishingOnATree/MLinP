__author__ = 'Rays'

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import os.path
from util_functions import util_functions as ut
import scipy.io as sio


# Neural network:
# input layer: 400 units
# hidden layer(1): 25 units
# output layer: 10 units
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10


def nn_cost_function(theta1, theta2, x, y, l):
    m, n = x.shape
    # convert y to 5000 * 10 [each col represents digit 0 - 9]
    identity_m = np.eye(num_labels)
    y_classifier = np.zeros((m, num_labels))  # 10 * 5000
    for i in range(m):
        k = y[i, 0] - 1
        y_classifier[i, :] = identity_m[k, :]

    # feedforword
    a1 = np.c_[np.ones((x.shape[0], 1)), x] # add intercept terms
    h1 = ut.sigmoid_function(a1.dot(theta1.T))
    a2 = np.c_[np.ones((h1.shape[0], 1)), h1] # add intercept terms
    # h2 = ut.sigmoid_function(a2.dot(theta2.T))
    # print(h2.shape)
    regularized_terms = l / 2.0 / m * ( sum(sum(np.power(theta1[:, 1:], 2))) +
                                        sum(sum(np.power(theta2[:, 1:], 2))) )
    return ut.logistic_cost_function(theta2.T, a2, y_classifier, 0) + regularized_terms

# load data
data = sio.loadmat('ex4data1.mat')
X = data['X']
y = data['y']
data_weights = sio.loadmat('ex4weights.mat')
theta1 = data_weights['Theta1']
theta2 = data_weights['Theta2']

l = 0
cost = nn_cost_function(theta1, theta2, X, y, l)
print(' Cost at initial theta (zeros) with lambda=0: %f ' % cost)

l = 1
cost = nn_cost_function(theta1, theta2, X, y, l)
print(' Cost at initial theta (zeros) with lambda=1: %f ' % cost)