#!/usr/bin/env python3
"""
Create a neuron
"""
import numpy as np


class Neuron:
    """Class that defines a single neuron for binary classification."""

    def __init__(self, nx):
        """
        Initialize the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx    
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

    @property
    def W(self):
        """Getter for the weights vector."""
        return self.__W

    @property
    def b(self):
        """Getter for the bias."""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output."""
        return self.__A
