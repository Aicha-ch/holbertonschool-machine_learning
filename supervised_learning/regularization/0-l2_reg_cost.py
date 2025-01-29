#!/usr/bin/env python3
"""
Calculate the cost of a neural network.
"""
import numpy as np


import numpy as np

def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculate the cost of a neural network
    """
    sum_of_squares = 0

    for l in range(1, L + 1):
        sum_of_squares += np.sum(np.square(weights[f'W{l}']))

    reg_cost = (lambtha / (2 * m)) * sum_of_squares
    total_cost = cost + reg_cost

    return total_cost
