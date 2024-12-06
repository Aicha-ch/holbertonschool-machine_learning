#!/usr/bin/env python3
"""
A function `matrix_transpose` that calculates
the transpose of a 2D matrix.
"""

def matrix_transpose(matrix):
    """
    Transposes a matrix (swapping rows and columns).

    Args:
        matrix (list of list): A 2D list (matrix) where each inner list represents a row.

    Returns:
        list of list: A new 2D list representing the transposed matrix. 
                      The rows of the original matrix become columns in the transposed matrix.
    """
    return [list(row) for row in zip(*matrix)]
