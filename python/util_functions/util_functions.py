__author__ = 'Rays'

import numpy as np


def sigmoid_function(z):
    return (1.0 / (1.0 + np.exp(-z))).reshape(z.shape)


def sigmoid_gradient_function(z):
    return sigmoid_function(z) * (1 - sigmoid_function(z))


def logistic_cost_function(theta, x, y, l, log_zero_replace=0.00000001):
    '''
    theta: n x 1 vector
    x: m x n matrix
    y: m x 1 vector
    '''
    m, n = x.shape
    h = sigmoid_function(np.dot(x, theta))
    lambda_cost_adj = l / 2 / m * np.power(theta, 2)
    lambda_cost_adj[0] = 0  # theta 0 should not be adjusted
    # substitue log(0) term with a very small positive constant as log(x) approaches -Inf when x approaches 0
    inner_term = 1 - h
    inner_term[inner_term == 0] = log_zero_replace
    inner_term = np.log(inner_term)
    diff_sum = np.multiply(y, np.log(h)) + np.multiply(1 - y, inner_term)
    return sum((sum(diff_sum) / -m + l / 2 / m * sum(np.power(lambda_cost_adj, 2))))
