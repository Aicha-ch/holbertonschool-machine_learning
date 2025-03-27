#!/usr/bin/env python3
"""
correlation matrix
"""

import numpy as np


def correlation(C):
    """
    correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std_devs = np.sqrt(np.diag(C))

    std_devs_outer = np.outer(std_devs, std_devs)

    correlation_matrix = C / std_devs_outer

    return correlation_matrix
