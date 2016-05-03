__author__ = 'Rays'

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import os.path
from util_functions import util_functions as ut
import random

input_layer_size  = 400;  # 20x20 Input Images of Digits
num_labels = 10;          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)
l = 0.7

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
        ops = {"maxiter": 10000}
        result = op.minimize(fun=ut.logistic_cost_function, x0=initial_theta, args=(x, classifier, l), method='TNC', jac=gradient_function, options=ops)
        print('%d result = %s' % (k+1, result.success))
        while not result.success:
            # try until success
            initial_theta = np.random.rand(x.shape[1], 1)
            result = op.minimize(fun=ut.logistic_cost_function, x0=initial_theta, args=(x, classifier, l), method='TNC', jac=gradient_function, options=ops)
            print('    Retry result = %s' % result.success)
        all_theta[k, :] = np.array(result.x).T
    print(all_theta.shape)
    return all_theta


def predict(theta, x):
    result = np.zeros((x.shape[0], 1))
    for m in range(x.shape[0]):
        case = x[m, :].reshape((x.shape[1], 1))
        prediction = ut.sigmoid_function(theta.dot(case))
        result[m, 0] = np.argmax(prediction) + 1
    return result


X = np.load('X.npy')
X = np.c_[np.ones((X.shape[0], 1)), X] # add intercept terms
y = np.load('y.npy')
theta1 = np.load('theta1.npy')
theta1 = np.load('theta2.npy')

sample = np.random.choice(X.shape[0], 20)
#show_images(sample)
if os.path.isfile('optimal_theta.npy'):
    optimal_theta = np.load('optimal_theta.npy')
    print('generating theta')
else:
    optimal_theta = one_vs_all(X, y, l)
    np.save('optimal_theta.npy', optimal_theta)
    print('using pre-calculated theta')
prediction = predict(optimal_theta, X)
print('Training accuracy = %f percent' % (sum((prediction==y) * 1)[0] * 100.0 / y.shape[0]))
