#!/usr/bin/env python3
"""
This script defines a Deep Neural Network for binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Define a deep neural network.
    """
    def __init__(self, nx, layers):
        """
        Initialize a deep neural network.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for layer_index in range(1, self.L + 1):
            layer_size = layers[layer_index - 1]
            prev_layer_size = nx if layer_index == 1 else layers[
                    layer_index - 2
                    ]
            self.weights[f'W{layer_index}'] = (
                    np.random.randn(layer_size, prev_layer_size) * np.sqrt(
                        2 / prev_layer_size
                        )
                    )
            self.weights[f'b{layer_index}'] = np.zeros((layer_size, 1))
