#!/usr/bin/env python3
"""
Calculate precision
"""
import numpy as np


def specificity(confusion):
    """
    Calculate precision
    """
    TP = np.diagonal(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - (TP + FN + FP)
    specificity = TN / (TN + FP)
    return specificity
