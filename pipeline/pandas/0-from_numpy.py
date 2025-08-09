#!/usr/bin/env python3
"""
Pandas DataFrame.
"""

import pandas as pd


def from_numpy(array):
    """
    Pandas DataFrame.
    """
    columns = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)
