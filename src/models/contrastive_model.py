import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_encoder(input_shape = (256, 256, 3)):
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )

    inputs = keras.Input(shape=input_shape)
    outputs = resnet(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="oct-encoder")
    return model

# encoder = create_encoder()

def add_projection_head(encoder, projection_units = 128):
    inputs = keras.Input(shape=(256, 256, 3))
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="oct-encoder_with_projection-head"
    )
    return model