__author__ = 'Rays'

import numpy as np


def sigmoid_function(z):
    return (1.0 / (1.0 + np.exp(-z))).reshape(z.shape)