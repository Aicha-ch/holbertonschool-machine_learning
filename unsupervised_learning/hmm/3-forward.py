#!/usr/bin/env python3
"""
The Forward Algorithm
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    F = np.zeros((N, T))

    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        F[:, t] = F[:, t-1] @ Transition * Emission[:, Observation[t]]

    P = np.sum(F[:, -1])

    return P, F
