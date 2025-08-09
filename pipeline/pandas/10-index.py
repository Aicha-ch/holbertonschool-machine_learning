#!/usr/bin/env python3
"""
Set the Timestamp colum.
"""


def index(df):
    """
    Set the Timestamp column.
    """
    df.set_index("Timestamp", inplace=True)
    return df
