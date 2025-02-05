#!/usr/bin/env python3
"""
Performing a convolution on grayscale images with custom padding.
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performing a convolution on grayscale images with custom padding.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    h_out = h + 2 * ph - kh + 1
    w_out = w + 2 * pw - kw + 1

    padded_images = np.pad(
            images, ((0, 0), (ph, ph), (pw, pw)), mode='constant',
            constant_values=0
            )

    output = np.zeros((m, h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            output[:, i, j] = np.sum(
                    padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
                    )

    return output
