#!/usr/bin/env python3
"""
Gradient Descent with Dropout.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Gradient Descent with Dropout.
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y

    for i in range(L, 0, -1):
        prev_A = cache[f"A{i - 1}"]

        dW = (1 / m) * np.matmul(dZ, prev_A.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA = np.matmul(weights[f"W{i}"].T, dZ)
            dA *= cache[f"D{i - 1}"]
            dA /= keep_prob
            dZ = dA * (1 - np.square(cache[f"A{i - 1}"]))

        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db
