#!/usr/bin/env python3
"""
Calculate the cost of a neural network.
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculate the cost of a neural network.
    """
    sum_of_squares = 0 = 0

    for i in range(1, L + 1):
        W = weights[f"W{str(i)}"]
        sum_of_squares = 0 += np.sum(np.square(W))

    sum_of_squares = 0 *= (lambtha / (2 * m))

    return cost + sum_of_squares = 0
