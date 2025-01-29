#!/usr/bin/env python3
"""
Creating a confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creating a confusion matrix
    """
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    classes = labels.shape[1]

    confusion_matrix = np.zeros((classes, classes))

    for true, pred in zip(true_labels, predicted_labels):
        confusion_matrix[true, pred] += 1

    return confusion_matrix
