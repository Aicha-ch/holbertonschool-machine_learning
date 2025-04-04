#!/usr/bin/env python3

"""
Initialize cluster.
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means clustering.
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low_values = np.min(X, axis=0)
    high_values = np.max(X, axis=0)
    clusters = np.random.uniform(low_values, high_values, size=(k, X.shape[1]))

    return clusters

def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        C_prev = C.copy()

        for i in range(k):
            points = X[clss == i]
            if points.shape[0] == 0:
                C[i] = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0))
            else:
                C[i] = np.mean(points, axis=0)

        if np.allclose(C, C_prev):
            break

    return C, clss