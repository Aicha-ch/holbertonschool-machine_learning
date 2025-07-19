#!/usr/bin/env python3
"""
Policy_gradient function.
"""

import numpy as np


def policy(matrix, weight):
    """
    Computing the policy
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=1, keepdims=True)
