__author__ = 'Rays'

import numpy as np


def sigmoid_function(z):
    return (1.0 / (1.0 + np.exp(-z))).reshape(z.shape)


def logistic_cost_function(theta, x, y, l):
    '''
    theta: n x 1 vector
    x: m x n matrix
    y: m x 1 vector
    '''
    m, n = x.shape
    theta = theta.reshape((n, 1))
    h = sigmoid_function(np.dot(x, theta))
    lambda_cost_adj = l / 2 / m * np.power(theta, 2)
    lambda_cost_adj[0] = 0  # theta 0 should not be adjusted
    diff_sum = np.multiply(y, np.log(h)) + np.multiply(np.add(1, np.negative(y)), np.log(np.add(1, np.negative(h))))
    return sum(diff_sum) / -m + l / 2 / m * sum(np.power(lambda_cost_adj, 2))