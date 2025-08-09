#!/usr/bin/env python3
"""
Sort a DataFrame in reverse.
"""


def flip_switch(df):
    """
    Sort a DataFrame in reverse.
    """
    sorted_df = df.sort_index(ascending=False)
    return sorted_df.transpose()
