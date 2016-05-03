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


def generate_y_classifier(m, y):
    # convert y to 5000 * 10 [each col represents digit 0 - 9]
    y_classifier = np.zeros((m, num_labels))  # 10 * 5000
    identity_m = np.eye(num_labels)
    for i in range(m):
        k = y[i, 0] - 1
        y_classifier[i, :] = identity_m[k, :]
    return y_classifier


def forward_feeding(theta1, theta2, x):
    a1 = np.c_[np.ones((x.shape[0], 1)), x]  # add intercept terms
    h2 = ut.sigmoid_function(a1.dot(theta1.T))
    a2 = np.c_[np.ones((h2.shape[0], 1)), h2]  # add intercept terms
    h3 = ut.sigmoid_function(a2.dot(theta2.T)) # h(x)
    return a1, a2, h2, h3


def nn_cost_function(nn_params, x, y, l):
    theta1, theta2 = unpack(nn_params)
    m, n = x.shape
    y_classifier = generate_y_classifier(m, y)

    # feedforword
    a1, a2, h2, h3 = forward_feeding(theta1, theta2, x)
    # print(h2.shape)
    regularized_terms = l / 2.0 / m * ( sum(sum(np.power(theta1[:, 1:], 2))) +
                                        sum(sum(np.power(theta2[:, 1:], 2))) )
    return ut.logistic_cost_function(theta2.T, a2, y_classifier, 0) + regularized_terms


def back_propogation(nn_params, x, y, l):
    theta1, theta2 = unpack(nn_params)
    m, n = x.shape
    y_classifier = generate_y_classifier(m, y)
    # feedforword
    a1, a2, h2, h3 = forward_feeding(theta1, theta2, x)
    #backpropogation
    delta3 = h3 - y_classifier
    delta2 = np.multiply(delta3.dot(theta2), np.multiply(a2, (1-a2)))
    delta2 = delta2[:, 1:] # removing the bias term
    #calculate theta1 gradient
    reg_term_theta1 = theta1 * float(l) / m
    reg_term_theta1[:, 0] = 0
    theta1_grad = delta2.T.dot(a1) / m + reg_term_theta1
    #calculate theta2 gradient
    reg_term_theta2 = theta2 * float(l) / m
    reg_term_theta2[:, 0] = 0
    theta2_grad = delta3.T.dot(a2) / m + reg_term_theta2
    return pack(theta1_grad, theta2_grad)


def pack(t1, t2):
    return np.r_[t1.ravel(), t2.ravel()]


def unpack(nn_params):
    t1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))].reshape(hidden_layer_size, (input_layer_size+1))
    t2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels, (hidden_layer_size+1))
    return t1, t2


def rand_initialize_weights(l_in, l_out):
    epsilon_init = 0.12
    t = np.random.rand(l_out, l_in + 1) * 2 * epsilon_init - epsilon_init
    return t


def predict(theta1, theta2, x):
    a1 = np.c_[np.ones((x.shape[0], 1)), x] # add intercept terms
    z2 = a1.dot(theta1.T)
    h2 = ut.sigmoid_function(z2)
    a2 = np.c_[np.ones((h2.shape[0], 1)), h2]
    z3 = a2.dot(theta2.T)
    h3 = ut.sigmoid_function(z3)
    return (np.argmax(h3, axis=1).reshape(x.shape[0], 1) + 1) # data converted from Octave where the index started from 1

# load data
data = sio.loadmat('ex4data1.mat')
X = data['X']
y = data['y']
data_weights = sio.loadmat('ex4weights.mat')
theta1 = data_weights['Theta1']
theta2 = data_weights['Theta2']

l = 0
cost = nn_cost_function(pack(theta1, theta2), X, y, l)
print(' Cost at initial theta (zeros) with lambda=0: %f ' % cost)

l = 1
cost = nn_cost_function(pack(theta1, theta2), X, y, l)
print(' Cost at initial theta (zeros) with lambda=1: %f ' % cost)

if os.path.isfile('optimal_theta1.npy'):
    optimal_theta1 = np.load('optimal_theta1.npy')
    optimal_theta2 = np.load('optimal_theta2.npy')
    print('using pre-calculated theta')
else:
    initial_theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    ops = {"maxiter": 500}
    result = op.minimize(fun=nn_cost_function, x0=pack(initial_theta1, initial_theta2),
                         args=(X, y, l), method='TNC', jac=back_propogation, options=ops)
    print(result)
    optimal_theta1, optimal_theta2 = unpack(result.x)
    np.save('optimal_theta1.npy', optimal_theta1)
    np.save('optimal_theta2.npy', optimal_theta2)
    print('generating theta')

prediction = predict(optimal_theta1, optimal_theta2, X)
print('Training accuracy = %f percent' % (sum((prediction==y) * 1)[0] * 100.0 / y.shape[0]))