#!/usr/bin/env python3
"""
Performing back propagation.
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performing back propagation.
    """
    m, h_new, w_new, c = dA.shape
    sh, sw = stride
    kh, kw = kernel_shape

    dA_prev = np.zeros(shape=A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw
                    if mode == 'avg':
                        avg_dA = dA[i, h, w, f] / kh / kw
                        dA_prev[i, v_start:v_end, h_start:h_end, f] +=\
                            (np.ones((kh, kw)) * avg_dA)
                    elif mode == 'max':
                        region = A_prev[i, v_start:v_end, h_start:h_end, f]
                        mask = (region == np.max(region))
                        dA_prev[i, v_start:v_end, h_start:h_end, f] +=\
                            mask * dA[i, h, w, f]

    return dA_prev
