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
