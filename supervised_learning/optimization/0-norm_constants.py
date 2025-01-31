#!/usr/bin/env python3
"""
Calculates the normalization.
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=0)

    return mean, std
