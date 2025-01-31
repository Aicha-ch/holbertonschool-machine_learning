#!/usr/bin/env python3
"""
Creates mini-batches
"""

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches
    """
    X, Y = shuffle_data(X, Y)

    m = X.shape[0]
    mi_batches = []

    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mi_batches.append((X_batch, Y_batch))

    return mi_batches
