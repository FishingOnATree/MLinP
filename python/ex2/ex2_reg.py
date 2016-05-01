__author__ = 'Rays'

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize as op

marker_size = 30


def plot_data(X, y):
    plt.scatter(X[(y == 1), 0], X[(y == 1), 1], marker='+', s=marker_size)
    plt.scatter(X[(y == 0), 0], X[(y == 0), 1], marker='o', color='y', s=marker_size)
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")
    plt.legend(['y=1', 'y=0'])
    plt.show()


def plot_decision_boundary(x1, x2, y, theta, lmda):
    x1 = x1.reshape(x1.size, 1)
    x2 = x2.reshape(x2.size, 1)
    plt.scatter(x1[y == 1], x2[y == 1], marker='+', s=marker_size)
    plt.scatter(x1[y == 0], x2[y == 0], marker='o', color='y', s=marker_size)

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros(shape=(len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = (map_features(np.array(u[i]), np.array(v[j])).dot(np.array(theta)))
    z = z.T
    plt.contour(u, v, z)
    plt.title('lambda = %f' % lmda)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
    plt.show()

def map_features(x1, x2):
    degree = 6
    out = np.ones((x1.size, 28))
    iteration = 0
    for i in range(1, degree + 1):
        for j in range(0, i+1):
            iteration += 1
            out[:, iteration] = np.power(x1, i-j) * np.power(x2, j)
    return out


def sigmoid_function(z):
    return (1.0 / (1.0 + np.exp(-z))).reshape(z.shape)


def cost_function_reg(theta, x, y, lmda):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    h = sigmoid_function(np.dot(x, theta))
    lambda_cost_adj = lmda / 2 / m * np.power(theta, 2)
    lambda_cost_adj[0] = 0  # theta 0 should not be adjusted
    diff_sum = np.multiply(y, np.log(h)) + np.multiply(np.add(1, np.negative(y)), np.log(np.add(1, np.negative(h))))
    return sum(diff_sum) / -m + lmda / 2 / m * sum(np.power(lambda_cost_adj, 2))


def gradient_function(theta, x, y, lmda):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    h = sigmoid_function(np.dot(x, theta)).reshape(y.shape)
    lambda_adj = np.multiply(theta, lmda/m).reshape(theta.shape)
    lambda_adj[0] = 0 #do not touch theta0
    return np.divide(np.dot(x.transpose(), np.subtract(h, y)), m) + lambda_adj


def predict(theta, x):
    m, n = x.shape
    h = sigmoid_function(np.dot(x, theta))
    out = np.zeros((m, 1))
    out[(h>=0.5)] = 1
    return out


data = np.genfromtxt("ex2data2.txt", delimiter=",")
X = data[:, 0:2].astype(float)
y = data[:, 2].astype(int)
#plot_data(X, y)

x = map_features(X[:, 0], X[:, 1])
y = y.reshape((len(y), 1))
m, n = x.shape # # of training cases/features
initial_theta = np.zeros((n, 1))
lmda = 1

## ============= Part 2: Regularization and Accuracies =============
ops = {"maxiter": 5000}
Result = op.minimize(fun=cost_function_reg, x0=initial_theta, args=(x, y, lmda), method='TNC', jac=gradient_function, options=ops)
#Result = op.fmin_tnc(cost_function_reg, initial_theta, fprime=gradient_function, args=(x, y, lmda))
#print(Result)
optimal_theta = Result.x
success_msg = Result.message
cost = cost_function_reg(optimal_theta, x, y, lmda)
print(success_msg, ' Cost at optimal theta (zeros): %f ' % cost)
plot_decision_boundary(X[:,0], X[:,1], y, optimal_theta, lmda)
p_res = predict(optimal_theta, x)
print('Training accuracy = %f percent' % (y[np.where(p_res == y)].size/float(y.size) * 100))