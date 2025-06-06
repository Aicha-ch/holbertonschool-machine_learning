#!/usr/bin/env python3
"""
Updating weights and bias
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updating weights and bias
    """
    m = Y.shape[1]
    A_prev = cache['A' + str(L - 1)]
    A_L = cache['A' + str(L)]

    dZ = A_L - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db

        if layer > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - A_prev ** 2)
