#!/usr/bin/env python3
"""
Variational Autoencoder
"""

import tensorflow.keras as keras


def sampling(args, latent_dims):
    """
    Sampling
    """
    z_mean, z_log_sigma = args
    epsilon = keras.backend.random_normal(
        shape=(keras.backend.shape(z_mean)[0],
               latent_dims))

    return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon


def build_encoder(input_dims, hidden_layers, latent_dims):
    """
    Builds the encoder part of the autoencoder.
    """
    encoder_input = keras.layers.Input(shape=(input_dims,))

    x = encoder_input
    for units in hidden_layers:
        x = keras.layers.Dense(units=units, activation='relu')(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    z = keras.layers.Lambda(lambda x: sampling(
        x, latent_dims))([mean, log_var])

    model_encoder = keras.Model(
        inputs=encoder_input, outputs=[mean, log_var, z])
    return model_encoder, mean, log_var


def build_decoder(latent_dims, hidden_layers, output_dims):
    """
    Building decoder
    """
    decoder_input = keras.layers.Input(shape=(latent_dims,))

    x = decoder_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units=units, activation='relu')(x)

    decoder_output = keras.layers.Dense(output_dims, activation='sigmoid')(x)

    return keras.models.Model(decoder_input, decoder_output)


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creating full variational autoencoder model.
    """
    encoder, mean, log_var = build_encoder(
        input_dims, hidden_layers, latent_dims)
    decoder = build_decoder(latent_dims, hidden_layers, input_dims)

    encoder_input = encoder.input
    z = encoder(encoder_input)[0]
    decoded_output = decoder(z)

    auto = keras.Model(inputs=encoder_input, outputs=decoded_output)

    reconstruction_loss = (
        keras.losses.binary_crossentropy(encoder_input,
                                         decoded_output))
    reconstruction_loss *= input_dims
    kl_loss = (1 + log_var - keras.backend.square(mean) -
               keras.backend.exp(log_var))
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
