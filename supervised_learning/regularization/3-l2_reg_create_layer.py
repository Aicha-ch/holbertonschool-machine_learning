#!/usr/bin/env python3
"""
Create a neural network layer with L2 regularization.
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creating a TensorFlow layer with L2 regularization.
    """
    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer
    )
    return layer(prev)
