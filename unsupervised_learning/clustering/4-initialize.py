#!/usr/bin/env python3
"""
Initialize
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    pi = np.full((k,), fill_value=1/k)

    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    S = np.tile(np.eye(X.shape[1]), (k, 1, 1))

    return pi, m, S
