#!/usr/bin/env python3
"""
Remove rows with NaN values.
"""


def prune(df):
    """
    Remove rows with NaN values.
    """
    return df[df["Close"].notna()]
