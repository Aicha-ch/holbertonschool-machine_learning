#!/usr/bin/env python3
"""
calculate the weighted moving average
"""


def moving_average(data, beta):
    """
    calculate the weighted moving average
    """
    if beta > 1 or beta < 0:
        return None
    vt = 0
    moving = []
    for i in range(len(data)):
        vt = beta * vt + (1 - beta) * data[i]
        correction = 1 - beta ** (i + 1)
        moving.append(vt / correction)
    return moving
