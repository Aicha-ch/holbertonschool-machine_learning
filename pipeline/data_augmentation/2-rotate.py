#!/usr/bin/env python3
"""
Rotate an image by 90 degrees
counter-clockwise.
"""

import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image.
    """
    return tf.image.rot90(image)
