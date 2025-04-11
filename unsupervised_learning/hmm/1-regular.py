#!/usr/bin/env python3
"""
Regular Markov Chain
"""

import numpy as np


def regular(P):
    """
    Determines the steady state probabilities
    of a regular Markov chain.
    """
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return None
    if P.ndim != 2:
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None

    n = P.shape[0]
    square_P = np.linalg.matrix_power(P, n**2)
    if not np.all((square_P > 0)):
        return None

    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    index = np.argmin(np.abs(eigenvalues - 1))
    steady_state = eigenvectors[:, index]

    steady_state = steady_state / np.sum(steady_state)
    return steady_state.reshape(1, n)
