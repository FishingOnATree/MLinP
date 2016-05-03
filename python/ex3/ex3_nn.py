__author__ = 'Rays'

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import os.path
from util_functions import util_functions as ut


# Neural network:
# input layer: 400 units
# hidden layer(1): 25 units
# output layer: 10 units
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;


def predict(theta1, theta2, x):
    a1 = np.c_[np.ones((x.shape[0], 1)), x] # add intercept terms
    z2 = a1.dot(theta1.T)
    h2 = ut.sigmoid_function(z2)
    a2 = np.c_[np.ones((h2.shape[0], 1)), h2]
    z3 = a2.dot(theta2.T)
    h3 = ut.sigmoid_function(z3)
    return (np.argmax(h3, axis=1).reshape(x.shape[0], 1) + 1) # data converted from Octave where the index started from 1



theta1 = np.load('theta1.npy') # 25*401
theta2 = np.load('theta2.npy') # 10*26
X = np.load('X.npy')
y = np.load('y.npy')
prediction = predict(theta1, theta2, X)
print('Training accuracy = %f percent' % (sum((prediction==y) * 1)[0] * 100.0 / y.shape[0]))