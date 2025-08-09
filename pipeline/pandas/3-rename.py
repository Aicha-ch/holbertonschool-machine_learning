#!/usr/bin/env python3
"""
Rename and modify a pandas DataFrame.
"""

import pandas as pd


def rename(df):
    """
    Rename and modify a pandas DataFrame.
    """
    df.rename(columns={"Timestamp": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
