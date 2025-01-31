#!/usr/bin/env python3
"""
momentum optimization algorithm with tensorflow
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    momentum optimization algorithm with tensorflow
    """
    optimize = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimize
