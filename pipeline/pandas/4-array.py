#!/usr/bin/env python3
"""
Extract data from a pandas DataFrame.
"""


def array(df):
    """
    Extract data from a pandas DataFrame.
    """
    selected_data = df[["High", "Close"]].tail(10)
    return selected_data.values
