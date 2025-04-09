#!/usr/bin/env python3
"""
Markov Chain
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain.
    """
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if len(P.shape) != 2 or len(s.shape) != 2:
        return None
    n, m = P.shape
    if n != m:
        return None
    if s.shape != (1, n):
        return None
    if not isinstance(t, int) or t < 1:
        return None

    state = s.copy()
    for _ in range(t):
        state = np.matmul(state, P)

    return state