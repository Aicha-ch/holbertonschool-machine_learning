#!/usr/bin/env python3
"""
Compute the Monte-Carlo policy gradient.
"""

import numpy as np


def policy(matrix, weight):
    """
    Computing the policy
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=1, keepdims=True)

def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and a
    weight matrix.
    """

    action_probs = policy(state, weight)

    action = np.random.choice(len(action_probs), p=action_probs)

    d_softmax = action_probs.copy()

    d_softmax[action] -= 1

    grad = -np.outer(state, d_softmax)

    return action, grad
