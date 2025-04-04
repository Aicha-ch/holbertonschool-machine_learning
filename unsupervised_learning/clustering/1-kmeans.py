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
        C_prev = np.copy(C)
        distances = np.sqrt(np.sum((X - C[:, np.newaxis]) ** 2, axis=2))
        clss = np.argmin(distances, axis=0)

        for i in range(k):
            cluster_mask = X[clss == i]
            if len(cluster_mask) == 0:
                C[i] = initialize(X, 1)
            else:
                C[i] = np.mean(X[clss == i], axis=0)

        distances = np.sqrt(np.sum((X - C[:, np.newaxis]) ** 2, axis=2))
        clss = np.argmin(distances, axis=0)

        if np.allclose(C, prev_ctds):
            break

    return C, clss