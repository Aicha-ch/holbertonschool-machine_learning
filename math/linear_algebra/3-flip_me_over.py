#!/usr/bin/env python3

def matrix_transpose(matrix):
    """
    Transposes a matrix.

    Args:
        matrix: A 2D list (matrix) to be transposed.
        Each inner list represents a row of the matrix.

    Returns:
        list of list: A new 2D list representing the transposed matrix.
        The rows of the matrix become columns in the transposed matrix.

    """
    return [list(row) for row in zip(*matrix)]
