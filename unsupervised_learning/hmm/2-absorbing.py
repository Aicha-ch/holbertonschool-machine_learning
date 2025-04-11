#!/usr/bin/env python3

import numpy as np

def absorbing(P):
    """
    Determines if a markov chain is absorbing.
    """
    if (not isinstance(P, np.ndarray) or P.ndim != 2 or
            P.shape[0] != P.shape[1]):
        return False

    n = P.shape[0]

    absorbing_states = np.where(np.diag(P) == 1)[0]

    if len(absorbing_states) == 0:
        return False

    non_absorbing_states = np.setdiff1d(np.arange(n), absorbing_states)

    if len(non_absorbing_states) == 0:
        return True

    Q = P[non_absorbing_states[:, None], non_absorbing_states]
    R = P[non_absorbing_states[:, None], absorbing_states]

    identity_matrix = np.eye(len(non_absorbing_states))
    try:
        N = np.linalg.inv(identity_matrix - Q)
        B = np.matmul(N, R)
    except np.linalg.LinAlgError:
        return False

    return np.all(np.any(B > 0, axis=1))