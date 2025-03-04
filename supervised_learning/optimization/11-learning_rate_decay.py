#!/usr/bin/env python3
"""
Updates the learning rate.
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate.
    """
    decayed_alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return decayed_alpha
