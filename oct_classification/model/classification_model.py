import tensorflow as tf


def get_classification_model(base_model,num_classes=4,dropout_rate=0.2,base_freeze_ratio=0.2):

    base_model.trainable = False
    fine_tune_at = int(len(base_model.layers) * (1.0 - 0.2))
    for layer in base_model.layers[100:]:
        layer.trainable = True
    # layers to add 
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # construct the model
    x = base_model.output
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    # outputs = prediction_layer(x)
    model = tf.keras.Model(base_model.inputs, outputs)
    return model
