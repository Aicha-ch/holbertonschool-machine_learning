#!/usr/bin/env python3
"""
Adjust the contrast of an image.
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Adjust the contrast of an image.
    """
    return tf.image.random_contrast(image, lower, upper)
