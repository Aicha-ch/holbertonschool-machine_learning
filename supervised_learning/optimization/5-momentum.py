#!/usr/bin/env python3
"""
Update variable using the gradient descent with momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Update variable using the gradient descent with momentum
    """
    momentum = beta1 * v + (1 - beta1) * grad
    updated_var = var - alpha * momentum
    return updated_var, momentum
