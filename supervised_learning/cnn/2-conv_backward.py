#!/usr/bin/env python3
"""
Performs back propagation
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0

    padded_A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    dA = np.zeros(shape=padded_A_prev.shape)
    dW = np.zeros(shape=W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw
                    dA[i, v_start:v_end, h_start:h_end, :] +=\
                        W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] +=\
                        padded_A_prev[i, v_start:v_end, h_start:h_end, :]\
                        * dZ[i, h, w, c]

    if padding == 'same':
        dA = dA[:, ph:-ph, pw:-pw, :]

    return dA, dW, db
