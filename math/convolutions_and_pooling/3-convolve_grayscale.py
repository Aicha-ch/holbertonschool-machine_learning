#!/usr/bin/env python3
"""
Strided Convolution
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Strided Convolution
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0

    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    padded_imgs = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                         mode='constant',)

    convolved = np.zeros((m, h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            region = padded_imgs[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            convolved[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved
