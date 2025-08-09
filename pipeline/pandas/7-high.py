#!/usr/bin/env python3
"""
Sort a pandas DataFrame.
"""


def high(df):
    """
    Sort the DataFrame.
    """
    return df.sort_values(by="High", ascending=False)
