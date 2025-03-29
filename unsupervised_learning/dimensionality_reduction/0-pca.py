#!/usr/bin/env python3
"""
PCA
"""

import numpy as np


def pca(X, var=0.95):
    """
    Perform PCA on the dataset X.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    cum_variance = np.cumsum(S ** 2) / np.sum(S ** 2)

    num_comp = np.argmax(cum_variance >= var) + 1

    return Vt[:num_comp + 1].T