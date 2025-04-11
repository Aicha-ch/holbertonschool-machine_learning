#!/usr/bin/env python3
"""
The Viretbi Algorithm
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    The Viretbi Algorithm
    """

    try:
        T = Observation.shape[0]
        N = Emission.shape[0]

        delta = np.zeros((N, T))
        psi = np.zeros((N, T), dtype=int)

        delta[:, 0] = Initial.T * Emission[:, Observation[0]]
        psi[:, 0] = 0

        for t in range(1, T):
            for j in range(N):
                prob = (delta[:, t-1] *
                        Transition[:, j] *
                        Emission[j, Observation[t]])
                delta[j, t] = np.max(prob)
                psi[j, t] = np.argmax(prob)

        P = np.max(delta[:, T-1])
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[:, T-1])

        for t in range(T-2, -1, -1):
            path[t] = psi[path[t+1], t+1]

        return path.tolist(), P

    except Exception:
        return None, None
