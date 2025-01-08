#!/usr/bin/env python3
"""
Defines a neural network.
"""
import numpy as np


class NeuralNetwork:
    """
    NeuralNetwork class.
    """

    def __init__(self, nx, nodes):
        """
        Initialize the neural network.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)  # Use internal attribute __W1
        self.__b1 = np.zeros((nodes, 1))        # Use internal attribute __b1
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @W1.setter
    def W1(self, value):
        self.__W1 = value

    @property
    def b1(self):
        return self.__b1

    @b1.setter
    def b1(self, value):
        self.__b1 = value

    @property
    def A1(self):
        return self.__A1

    @A1.setter
    def A1(self, value):
        self.__A1 = value

    @property
    def W2(self):
        return self.__W2

    @W2.setter
    def W2(self, value):
        self.__W2 = value

    @property
    def b2(self):
        return self.__b2

    @b2.setter
    def b2(self, value):
        self.__b2 = value

    @property
    def A2(self):
        return self.__A2

    @A2.setter
    def A2(self, value):
        self.__A2 = value
