#!/usr/bin/env python3
"""
performing a valid convolution on grayscale images
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performing a valid convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    h_out = h - kh + 1
    w_out = w - kw + 1

    output = np.zeros((m, h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            output[:, i, j] = np.sum(
                    images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
                    )

    return output
