#!/usr/bin/env python3
"""
PCA on a dataset
"""
import numpy as np


def pca(X, ndim):
    """
    PCA on a dataset
    """
    X_centered = X - np.mean(X, axis=0)

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    W = Vt[:ndim].T

    T = np.dot(X_centered, W)

    return T