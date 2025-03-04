#!/usr/bin/env python3
"""
Normalize an unactivated output.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output.
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)

    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    return gamma * Z_norm + beta
