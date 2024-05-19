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

        if config.MODEL.type == "feature_extractor":
            return arch_func(
                include_top=False,
                weights='imagenet' if config.MODEL.backbone_pretrained else None,
                input_shape=(config.DATA.image_size, config.DATA.image_size, 3),
                pooling='avg'
            )
        # elif config.MODEL.type == "classification":
        #     # get base model
        #     IMG_SHAPE = (180,180,3)
        #     base_model =tf.keras.applications.EfficientNetB4(input_shape=IMG_SHAPE,
        #                                                 include_top=False,
        #                                                 weights='imagenet')
        else:
            return arch_func(
                include_top=False,
                weights='imagenet' if config.MODEL.backbone_pretrained else None,
                input_shape=(config.DATA.image_size, config.DATA.image_size, 3)
            )

    def model_func():
        from models.embedding_model import EmbeddingModel
        from models.classification_model import get_classification_model

        if config.MODEL.type == 'embedding_model':
            # embedding model
            # print(config)
            # print(backbone_func())
            embedding_model = EmbeddingModel(
                arch=backbone_func(),
                emb_size=config.MODEL.emb_size,
                arch_trainable=config.MODEL.pretrained_model_fully_trainable,
                dropout_rate=config.MODEL.dropout_rate,
                l2_regularization=config.MODEL.l2_regularization
                )
            return embedding_model
        elif config.MODEL.type == 'feature_extractor':
            return backbone_func()
        elif config.MODEL.type == 'classification':
            return get_classification_model(backbone_func(),num_classes=config.MODEL.num_classes)
        else:
            raise Exception(f'{config.MODEL.type} is not yet defined properly')

    return model_func

# model = get_base_model(config)


# import importlib
# from src.models.encoder import get_encoder

# __all__ =[
#     'get_model'
# ]

# def get_model(input, model_name, model_params, train_mode, is_train):
#     model_module = importlib.import_module("src.models." + model_name, package=None)
#     encoder = get_encoder(input=input, encoder_name=model_params["encoder"],
#                           pretrained=is_train)
    
#     return model_module.SmpMultitaskModel(
#         input=input, encoder_module=encoder, 
#         **model_params, train_mode=train_mode, 
#         is_train=is_train)