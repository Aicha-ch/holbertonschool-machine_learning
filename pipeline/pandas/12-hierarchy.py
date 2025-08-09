#!/usr/bin/env python3
"""
Concatenate two pandas DataFrames.
"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenate two pandas DataFrames.
    """
    df1 = index(df1)
    df2 = index(df2)

    start, end = 1417411980, 1417417980
    df1_filtered = df1[(df1.index >= start) & (df1.index <= end)]
    df2_filtered = df2[(df2.index >= start) & (df2.index <= end)]

    concat = pd.concat([df2_filtered, df1_filtered], keys=["bitstamp",
                                                                 "coinbase"])
    concat = concat.swaplevel(0, 1)
    concat = concat.sort_index(level=0)

    return concat
