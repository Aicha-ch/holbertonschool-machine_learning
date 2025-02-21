#!/usr/bin/env python3
""" CNN to classify the CIFAR 10 dataset
"""

from tensorflow import keras as K
import numpy as np


def preprocess_data(X, Y):
    """Preprocess the CIFAR-10 data."""
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == "__main__":
    # Load the CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Define input shape and resize layer
    input_shape = (32, 32, 3)
    new_shape = (224, 224)
    
    # Correction : Relier la sortie de Resizing à DenseNet121
    inputs = K.layers.Input(shape=input_shape)
    resized = K.layers.Resizing(new_shape[0], new_shape[1])(inputs)

    # Load the DenseNet121 model with pre-trained weights
    base_model = K.applications.DenseNet121(weights='imagenet', include_top=False, input_tensor=resized)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom top layers for CIFAR-10 classification
    x = K.layers.GlobalAveragePooling2D()(base_model.output)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    #  Associate inputs with model inputs
    model = K.models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=128)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, Y_test)

    # Print the evaluation results
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Save the model
    model.save('cifar10.h5')
