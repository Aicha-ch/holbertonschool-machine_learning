#!/usr/bin/env python3
"""
loading data from a file.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file.
    """
    return pd.read_csv(filename, delimiter=delimiter)
