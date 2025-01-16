#!/usr/bin/env python3
"""
Create the forward propagation graph.
"""

import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.
    """
    next_input = x

    for size, activ in zip(layer_sizes, activations):
        next_input = create_layer(next_input, size, activ)

    return next_input
