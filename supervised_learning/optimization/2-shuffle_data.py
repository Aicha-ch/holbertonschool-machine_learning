#!/usr/bin/env python3
"""
shuffle data points
"""
import numpy as np


def shuffle_data(X, Y):
    """
    shuffle data points
    """
    permut = np.random.permutation(X.shape[0])
    return X[permut, :], Y[permut, :]
