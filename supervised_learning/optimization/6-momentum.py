#!/usr/bin/env python3
"""
momentum optimization algorithm with tensorflow
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    momentum optimization algorithm with tensorflow
    """
    return tf.train.MomentumOptimizer(alpha, momentum=beta1).minimize(loss)
