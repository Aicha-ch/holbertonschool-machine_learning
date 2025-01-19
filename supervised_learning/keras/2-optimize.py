#!/usr/bin/env python3

"""
set up Adam optimization for a keras model.
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    sets up Adam optimization for a keras model
    """
    opt_adam = K.optimizers.Adam(learning_rate=alpha,
                                 beta_1=beta1,
                                 beta_2=beta2)

    network.compile(optimizer=opt_adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return None
