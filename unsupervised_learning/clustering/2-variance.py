#!/usr/bin/env python3
"""
Variance
"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    distances = np.min(np.linalg.norm(X[:, np.newaxis] - C, axis=-1), axis=-1)

    return np.sum(distances ** 2)
