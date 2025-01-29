#!/usr/bin/env python3
"""
Calculate score
"""
import numpy as np


def f1_score(confusion):
    """
    calculate score
    """
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    prec_value = precision(confusion)
    sens_value = sensitivity(confusion)
    F1 = 2 * (prec_value * sens_value) / (prec_value + sens_value)
    return F1
