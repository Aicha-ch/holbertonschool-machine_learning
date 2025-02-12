#!/usr/bin/env python3
"""
forward propagation
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    forward propagation
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    new_h = (h_prev - kh) // sh + 1
    new_w = (w_prev - kw) // sw + 1

    if mode == 'max':
        pooling_func = np.max
    elif mode == 'avg':
        pooling_func = np.mean

    output = np.zeros((m, new_h, new_w, c_prev))

    for i in range(new_h):
        for j in range(new_w):
            region = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            output[:, i, j, :] = pooling_func(region, axis=(1, 2))

    return output
