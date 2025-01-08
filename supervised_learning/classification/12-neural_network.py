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

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = np.zeros((1, 1))
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for the weights of the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Getter for the bias of the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Getter for the activated output of the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Getter for the weights of the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Getter for the bias of the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Getter for the activated output of the output neuron (prediction).
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation.
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost
