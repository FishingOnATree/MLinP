__author__ = 'Rays'

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize as op


def plot_data(X, y):
    plt.scatter(X[(y == 1), 0], X[(y == 1), 1], marker='+', s=40)
    plt.scatter(X[(y == 0), 0], X[(y == 0), 1], marker='o', color='y', s=40)
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")
    plt.legend(['y=1', 'y=0'])
    plt.show()


def map_features(x1, x2):
    degree = 6
    out = np.ones((len(x1), 28))
    iteration = 0
    for i in range(1, degree + 1):
        for j in range(0, i+1):
            iteration += 1
            out[:, iteration] = np.power(x1, i-j) * np.power(x2, j)
    return out


def sigmoid_function(z):
    return 1.0 / (1.0 + np.exp(-z))

def cost_function_reg(theta, x, y, lmda):
    m = len(y)
    grad = np.zeros(initial_theta.shape)
    h = sigmoid_function(np.dot(x, theta))
    lambda_cost_adj = lmda / 2 / m * np.power(theta, 2)
    lambda_cost_adj[0, 0] = 0  # theta 0 should not be adjusted
    diff_sum = np.multiply(y, np.log(h)) + np.multiply(np.add(1, np.negative(y)), np.log(np.add(1, np.negative(h))))
    return sum(diff_sum) / -m + lmda / 2 / m * sum(np.power(lambda_cost_adj, 2))


def gradient_function(theta, X, y, lmda):
    h = sigmoid_function(np.dot(x, theta))
    lambda_adj = np.multiply(theta, lmda/m)
    lambda_adj[0, 0] = 0 #do not touch theta0
    return np.divide(np.dot(x.transpose(), np.subtract(h, y)), m) + lambda_adj


data = np.genfromtxt("ex2data2.txt", delimiter=",")
X = data[:, 0:2].astype(float)
y = data[:, 2].astype(int)
# plot_data(X, y)

x = map_features(X[:, 0], X[:, 1])
y = y.reshape((len(y), 1))
m, n = x.shape # # of training cases/features
initial_theta = np.zeros((n, 1))
lmda = 1
cost = cost_function_reg(initial_theta, x, y, lmda)
print('Cost at initial theta (zeros): %f\n' % cost)

## ============= Part 2: Regularization and Accuracies =============
#Result = op.minimize(fun=cost_function_reg, x0=initial_theta, args=(X, y, lmda), method='TNC', jac=gradient_function)
#optimal_theta = Result.x



