#!/usr/bin/env python3
"""
calculate sensivity
"""
import numpy as np


def sensitivity(confusion):
    """
    calculate sensivity
    """
    TP = np.diagonal(confusion)
    P = np.sum(confusion, axis=1)
    return TP / P
