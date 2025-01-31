#!/usr/bin/env python3
"""
Adam optimization algorithm.
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Adam optimization algorithm.
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    v_updated = v / (1 - beta1 ** t)
    s_updated = s / (1 - beta2 ** t)

    _var = var - alpha * v_updated / (np.sqrt(s_updated) + epsilon)

    return _var, v, s
