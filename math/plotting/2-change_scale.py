#!/usr/bin/env python3
"""
This function generates a line graph
of exponential Decay of C-14
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    This function generates a line graph
    of exponential Decay of C-14
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y, label='C-14 decay')
    plt.title("Exponential Decay of C-14")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.xlim(0, 28600)
    plt.yscale("log")
    plt.show()
