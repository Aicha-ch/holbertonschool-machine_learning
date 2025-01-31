#!/usr/bin/env python3
"""
Update variable using the RMSProp
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Update variable using the RMSProp
    """
    s = beta2 * s + (1 - beta2) * np.square(grad)
    updated_var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return updated_var, s
