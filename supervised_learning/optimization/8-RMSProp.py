#!/usr/bin/env python3
"""
Set up the RMSProp optimization
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Set up the RMSProp optimization
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=beta2,
                                            epsilon=epsilon)
    return optimizer
