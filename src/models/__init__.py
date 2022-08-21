import tensorflow as tf
import os

"""
To use this interface, add new models to get_model's if-elif-statement and follow this model interface:
"""

__all__ = [
    'get_model_func',
    'model_func',
]


def get_model_func(config, pretrained_dir=None, pretrained_h5=None):
    """
    Returns model function for the model specified in config.
    Model func returns two models, training_model and embedding model for inference
    """

    def backbone_func():
        arch_func = None
        if config.MODEL.backbone_arch == 'mobilenetv2':
            arch_func = tf.keras.applications.MobileNetV2
        elif config.MODEL.backbone_arch == 'resnet50':
            arch_func = tf.keras.applications.ResNet50
        elif config.MODEL.backbone_arch == 'effnetb0':
            arch_func = tf.keras.applications.EfficientNetB0
        elif config.MODEL.backbone_arch == 'effnetb1':
            arch_func = tf.keras.applications.EfficientNetB1
        elif config.MODEL.backbone_arch == 'effnetb2':
            arch_func = tf.keras.applications.EfficientNetB2
        elif config.MODEL.backbone_arch == 'effnetb3':
            arch_func = tf.keras.applications.EfficientNetB3
        elif config.MODEL.backbone_arch == 'effnetb4':
            arch_func = tf.keras.applications.EfficientNetB4
        elif config.MODEL.backbone_arch == 'effnetb5':
            arch_func = tf.keras.applications.EfficientNetB5
        elif config.MODEL.backbone_arch == 'effnetb6':
            arch_func = tf.keras.applications.EfficientNetB6
        elif config.MODEL.backbone_arch == 'effnetb7':
            arch_func = tf.keras.applications.EfficientNetB7
        elif config.MODEL.backbone_arch == 'densenet121':
            arch_func = tf.keras.applications.DenseNet121
        elif config.MODEL.backbone_arch == 'xception':
            arch_func = tf.keras.applications.Xception
        elif config.MODEL.backbone_arch == 'resnet101':
            arch_func = tf.keras.applications.ResNet101
        else:
            raise Exception(f'{config.MODEL.backbone_arch} is not yet implemented')

        # if config.MODEL.type == "feature_extractor":
        #     return arch_func(
        #         include_top=False,
        #         weights='imagenet' if config.MODEL.backbone_pretrained else None,
        #         input_shape=(config.DATA.image_size, config.DATA.image_size, 3),
        #         pooling='avg'
        #     )
        # # elif config.MODEL.type == "classification":
        # #     # get base model
        # #     IMG_SHAPE = (180,180,3)
        # #     base_model =tf.keras.applications.EfficientNetB4(input_shape=IMG_SHAPE,
        # #                                                 include_top=False,
        # #                                                 weights='imagenet')
        # else:
        #     return arch_func(
        #         include_top=False,
        #         weights='imagenet' if config.MODEL.backbone_pretrained else None,
        #         input_shape=(config.DATA.image_size, config.DATA.image_size, 3)
        #     )

    def model_func():
        from src.models.contrastive_model import create_encoder, add_projection_head
        print(config.MODEL.type)
        if config.MODEL.type == 'constrastive_model':
            encoder = create_encoder()
            encoder_with_projection_head  = add_projection_head(encoder, projection_units = 128)
            return encoder_with_projection_head
        elif config.MODEL.type == 'autoencoder':
            return backbone_func()
        else:
            raise Exception(f'{config.MODEL.type} is not yet defined properly')

    return model_func