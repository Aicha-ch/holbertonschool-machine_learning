#!/usr/bin/env python3
"""
Normalize a matrix
"""


def normalize(X, m, s):
    """
    Normalize a matrix
    """
    return (X - m) / s
