#!/usr/bin/env python3

"""
Building a neural network with Keras.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Building a neural network with Keras.
    """
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i, layer in enumerate(layers):
        x = K.layers.Dense(
            layer,
            activation=activations[i],
            kernel_regularizer=K.regularizers.L2(lambtha)
        )(x)

        if i != len(layers) - 1 and keep_prob is not None:
            x = (K.layers.Dropout(1 - keep_prob))(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
