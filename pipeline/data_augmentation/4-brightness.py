#!/usr/bin/env python3
"""
Changes the brightness of an image.
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Changes the brightness of an image.
    """
    return tf.image.random_brightness(image, max_delta)
