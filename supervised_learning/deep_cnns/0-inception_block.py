#!/usr/bin/env python3
"""
Building an inception block.
"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Building an inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv_F1 = K.layers.Conv2D(F1, kernel_size=(1, 1), padding='same',
                              activation='relu')(A_prev)

    conv3x3_reduce = K.layers.Conv2D(F3R, kernel_size=(1, 1), padding='same',
                                     activation='relu')(A_prev)
    conv_F3 = K.layers.Conv2D(F3, kernel_size=(3, 3), padding='same',
                              activation='relu')(conv3x3_reduce)

    conv5x5_reduce = K.layers.Conv2D(F5R, kernel_size=(1, 1), padding='same',
                                     activation='relu')(A_prev)
    conv_F5 = K.layers.Conv2D(F5, kernel_size=(5, 5), padding='same',
                              activation='relu')(conv5x5_reduce)

    maxpool = K.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1),
                                 padding='same')(A_prev)
    maxpool_conv_F1 = K.layers.Conv2D(FPP, kernel_size=(1, 1), padding='same',
                                      activation='relu')(maxpool)

    inception_output = K.layers.Concatenate(
        axis=-1)([conv_F1, conv_F3, conv-F5, maxpool_conv_F1])

    return inception_output
