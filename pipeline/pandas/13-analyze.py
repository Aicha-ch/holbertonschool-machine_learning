#!/usr/bin/env python3
"""
Compute descriptive statistics.
"""


def analyze(df):
    """
    Compute descriptive statistics.
    """
    stats = df.loc[:, df.columns != "Timestamp"].describe()
    return stats
