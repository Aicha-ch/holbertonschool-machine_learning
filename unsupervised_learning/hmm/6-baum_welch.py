#!/usr/bin/env python3
"""
The Baum-Welch Algorithm
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Forward algorithm for a hidden Markov model.
    """
    T = Observation.shape[0]
    N = Emission.shape[0]

    F = np.zeros((N, T))

    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(
                F[:, t-1] * Transition[:, j] * Emission[j, Observation[t]]
            )

    return F


def backward(Observation, Emission, Transition):
    """
    backward algorithm for a hidden Markov model.
    """
    T = Observation.shape[0]
    N = Emission.shape[0]

    B = np.zeros((N, T))

    B[:, T-1] = 1

    for t in range(T-2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                Transition[i, :] * Emission[:, Observation[t+1]] * B[:, t+1]
            )

    return B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Baum-Welch algorithm to estimate the parameters of
    a hidden Markov model.
    """
    N, M = Emission.shape
    T = Observations.shape[0]

    for n in range(iterations):
        alpha = forward(Observations, Emission, Transition, Initial)
        beta = backward(Observations, Emission, Transition)

        xi = np.zeros((N, N, T-1))
        gamma = np.zeros((N, T))

        for t in range(T-1):
            denominator = np.sum(alpha[:, t] * beta[:, t])
            for i in range(N):
                gamma[i, t] = (alpha[i, t] * beta[i, t]) / denominator
                for j in range(N):
                    xi[i, j, t] = (alpha[i, t] * Transition[i, j] *
                                   Emission[j, Observations[t+1]] *
                                   beta[j, t+1]) / denominator

        gamma[:, T-1] = (alpha[:, T-1] * beta[:, T-1]) / np.sum(
            alpha[:, T-1] * beta[:, T-1]
        )

        Transition = np.sum(xi, axis=2) / np.sum(
            gamma[:, :-1], axis=1
        ).reshape(-1, 1)

        for k in range(M):
            Emission[:, k] = np.sum(gamma[:, Observations == k], axis=1)
        Emission /= np.sum(gamma, axis=1).reshape(-1, 1)

    return Transition, Emission
