#!/usr/bin/env python3
"""
Slice specific columns and rows from a pandas DataFrame.
"""


def slice(df):
    """
    Slice specific columns and rows from a pandas DataFrame.
    """
    sliced_df = df[["High", "Low", "Close", "Volume_(BTC)"]]
    return sliced_df.iloc[::60]
