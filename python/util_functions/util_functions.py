__author__ = 'Rays'

import numpy as np


def sigmoid_function(z):
    return (1.0 / (1.0 + np.exp(-z))).reshape(z.shape)


def logistic_cost_function(theta, x, y, l, neg_inf_replace= -10000):
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
    # substitue -Inf with a large negative constant
    inner_term = np.log(1 - h)
    inner_term[np.isneginf(inner_term)] = neg_inf_replace
    diff_sum = np.multiply(y, np.log(h)) + np.multiply(1 - y, inner_term)
    return (sum(diff_sum) / -m + l / 2 / m * sum(np.power(lambda_cost_adj, 2)))[0]