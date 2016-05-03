__author__ = 'Rays'

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from util_functions import util_functions as ut

input_layer_size  = 400;  # 20x20 Input Images of Digits
num_labels = 10;          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)
l = 0.1

def show_images(sample):
    plt.imshow(X[sample, 1:].reshape(-1, 20).T)
    plt.axis('off')
    plt.show()


def gradient_function(theta, x, y, lmda):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    h = ut.sigmoid_function(np.dot(x, theta)).reshape(y.shape)
    lambda_adj = np.multiply(theta, lmda/m).reshape(theta.shape)
    lambda_adj[0] = 0 #do not touch theta0
    return np.divide(np.dot(x.transpose(), np.subtract(h, y)), m) + lambda_adj


def one_vs_all(x, y, l):
    initial_theta = np.random.rand(x.shape[1], 1)
    all_theta = np.zeros((num_labels, x.shape[1]))
    for k in range(num_labels):
        classifier = (y == (k+1)) * 1
        # cost = ut.logistic_cost_function(initial_theta, x, classifier, l)
        # print("Initial cost = %f" % cost)
        ops = {"maxiter": 5000}
        result = op.minimize(fun=ut.logistic_cost_function, x0=initial_theta, args=(x, y, l), method='TNC', jac=gradient_function, options=ops)
        print('%d result = %s' % (k+1, result.success))
    print(all_theta.shape)
    return all_theta


X = np.load('X.npy')
X = np.c_[np.ones((X.shape[0], 1)), X] # add intercept terms
y = np.load('y.npy')
theta1 = np.load('theta1.npy')
theta1 = np.load('theta2.npy')

sample = np.random.choice(X.shape[0], 20)
#show_images(sample)
one_vs_all(X, y, l)
