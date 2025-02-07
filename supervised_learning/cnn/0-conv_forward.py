#!/usr/bin/env python3
"""
Propagation over a convolutional layer of a neural network.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Propagation over a convolutional layer of a neural network.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    pad_h, pad_w = (0, 0)
    if padding == "same":
        pad_h = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pad_w = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))

    padded = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    mode="constant", constant_values=(0, 0))

    new_h = (h_prev + 2 * pad_h - kh) // sh + 1
    new_w = (w_prev + 2 * pad_w - kw) // sw + 1
    output = np.zeros((m, new_h, new_w, c_new))

    for row in range(new_h):
        for col in range(new_w):
            for ch in range(c_new):
                slice_A = padded[:, row * sh:row * sh + kh, col * sw:col * sw
                                 + kw]
                slice_A_sum = np.sum(slice_A * W[:, :, :, ch], axis=(1, 2, 3))
                output[:, row, col, ch] = slice_A_sum
    return activation(output + b)
