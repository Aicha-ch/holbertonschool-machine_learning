#!/usr/bin/env python3

"""
Train a model using mini-batch gradient descent.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Train a model using mini-batch gradient descent.
    """
    callback = []
    if early_stopping is True and validation_data is not None:
        callback = K.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=patience)

    return network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=[callback])
