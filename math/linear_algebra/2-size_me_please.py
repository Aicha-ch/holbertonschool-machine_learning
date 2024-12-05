#!/usr/bin/env python3

"""
This module defines a function matrix_shape that calculates the shape
of a matrix
"""


def matrix_shape(matrix):
    if not isinstance(matrix, list):
        return []
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
