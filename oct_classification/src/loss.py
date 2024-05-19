import os
import tensorflow as tf
def get_loss_dict(config):

    loss_dict = {
        tf.keras.losses.CategoricalCrossentropy()
        }

    return loss_dict