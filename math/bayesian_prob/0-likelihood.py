#!/usr/bin/env python3
"""
Calculates the likelihood.
"""

import numpy as np


def binomial_coefficient(n, k):
    """Calculate binomial coefficient using factorial from numpy."""
    return np.math.factorial(n) // (np.math.factorial(k) * np.math.factorial(n - k))


def likelihood(x, n, P):
    """
    Calculates the likelihood.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the likelihood
    binomial_coeff = binomial_coefficient(n, x)
    likelihoods = binomial_coeff * np.power(P, x) * np.power((1 - P), (n - x))

    return likelihoods
