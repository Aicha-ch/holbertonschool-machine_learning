#!/usr/bin/env python3
"""
Find the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    A function that calculates the derivate of a polynomial
    """
    if not isinstance(poly, list) or any(
            not isinstance(coef, (int, float)) for coef in poly
            ):
        return None

    if len(poly) == 0:
        return None
    if len(poly) < 2:
        return [0]

    derivative = [i * poly[i] for i in range(1, len(poly))]

    if not derivative:
        return [0]

    return derivative
