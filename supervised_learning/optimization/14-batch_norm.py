#!/usr/bin/env python3
"""
Creates a batch normalization layer.
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense_layer = tf.keras.layers.Dense(units=n,
                                        kernel_initializer=initializer)

    Z = dense_layer(prev)

    gamma = tf.Variable(initial_value=tf.ones((1, n)), name='gamma')
    beta = tf.Variable(initial_value=tf.zeros((1, n)), name='beta')

    epsilon = 1e-7

    mean, variance = tf.nn.moments(Z, axes=[0])

    Z_batch_norm = tf.nn.batch_normalization(
        x=Z,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon)

    return activation(Z_batch_norm)
