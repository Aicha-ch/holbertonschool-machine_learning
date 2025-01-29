#!/usr/bin/env python3

"""Calculate the cost of a neural network"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculate the cost of a neural network
    """
    l2_cost = 0
    for layer_idx in range(L):
        l2_cost += np.sum(weights.get("W{}".format(layer_idx + 1))**2)

    return cost + (lambtha / (2 * m)) * l2_cost
