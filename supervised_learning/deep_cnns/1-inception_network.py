#!/usr/bin/env python3
"""
Builds the Inception network as described in
'Going Deeper with Convolutions (2014)'.
"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the Inception network as described in
    'Going Deeper with Convolutions (2014)'.
    """
    inputs = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding="same",
                            activation="relu")(inputs)

    pool1 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                   padding="same")(conv1)

    conv2 = K.layers.Conv2D(filters=64,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding="same",
                            activation="relu")(pool1)
    conv3 = K.layers.Conv2D(filters=192,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            activation="relu")(conv2)

    pool2 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                   padding="same")(conv3)

    inception_3a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])

    pool3 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                   padding="same")(inception_3b)

    inception_4a = inception_block(maxpool_3, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])

    pool4 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                   padding="same")(inception_4e)

    inception_5a = inception_block(maxpool_4, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AvgPool2D(pool_size=(7, 7),
                                  strides=(1, 1))(inception_5b)

    dropout = K.layers.Dropout(rate=0.4)(avg_pool)

    outputs = K.layers.Dense(units=1000, activation="softmax")(dropout)

    return K.Model(inputs=inputs, outputs=outputs)
