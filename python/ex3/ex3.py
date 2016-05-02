__author__ = 'Rays'

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

input_layer_size  = 400;  # 20x20 Input Images of Digits
num_labels = 10;          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)


def show_images(sample):
    plt.imshow(X[sample, 1:].reshape(-1, 20).T)
    plt.axis('off')
    plt.show()


X = np.load('X.npy')
X = np.c_[np.ones((X.shape[0], 1)), X] # add intercept terms
y = np.load('y.npy')
theta1 = np.load('theta1.npy')
theta1 = np.load('theta2.npy')

sample = np.random.choice(X.shape[0], 20)
show_images(sample)