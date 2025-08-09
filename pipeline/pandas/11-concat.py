#!/usr/bin/env python3
"""
Concatenate two pandas DataFrames.
"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenate two pandas DataFrames.
    """
    df1 = index(df1)
    df2 = index(df2)

    df2_filtered = df2[df2.index <= 1417411920]

    concat = pd.concat([df2_filtered, df1], keys=["bitstamp", "coinbase"])

    return concat
