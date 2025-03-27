#!/usr/bin/env python3
"""
multivariate normal distribution.
"""

import numpy as np


class MultiNormal:
    """
    multivariate normal distribution.
    """
    def __init__(self, data):
        """
        multivariate normal distribution.
        """
        # check if data is a 2D numpy array
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        data_centered = data - self.mean
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)
