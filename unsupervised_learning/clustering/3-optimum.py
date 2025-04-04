#!/usr/bin/env python3
"""
Optimizing k - Kmeans
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if isinstance(kmax, int) and kmax <= kmin:
        return None, None

    if kmax is None:
        max_clusters = X.shape[0]
    else:
        max_clusters = kmax

    results = []
    d_vars = []

    C, clss = kmeans(X, kmin, iterations)
    base_variance = variance(X, C)
    results.append((C, clss))
    d_vars = [0.0]

    k = kmin + 1
    while k < max_clusters + 1:
        C, clss = kmeans(X, k, iterations)
        current_variance = variance(X, C)
        results.append((C, clss))
        d_vars.append(base_variance - current_variance)
        k += 1

    return results, d_vars
