import tensorflow as tf
import os

def add_validation_callbacks(config,callbacks):
    if config.TRAIN.early_stopping_epochs > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=config.TRAIN.early_stopping_epochs))
    if config.TRAIN.reduce_lr_on_plateau_patience >= 0:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=config.TRAIN.reduce_lr_on_plateau_patience))
    return callbacks

def get_callbacks(config, save_dir, has_validation):
    callbacks = []
    if has_validation: # callbacks that require val_loss
        callbacks = add_validation_callbacks(config, callbacks)

    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(save_dir, 'logs'), profile_batch = 2
    ))

    return callbacks