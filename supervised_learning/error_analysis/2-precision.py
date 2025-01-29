#!/usr/bin/env python3
"""
calculate precision
"""
import numpy as np


def precision(confusion):
    """
    calculate precision
    """
    TP = np.diagonal(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    precision = TP / predicted_positives
    return precision
