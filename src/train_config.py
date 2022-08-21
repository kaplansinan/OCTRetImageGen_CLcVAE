from easydict import EasyDict as edict
import numpy as np

config = edict()
config.TRAIN = edict()
config.MODEL = edict()
config.DATA = edict()

#####################################################
# TRAIN
#####################################################
# Unfreeze the base model
config.TRAIN.base_model_trainable = True

#####################################################
# MODEL
#####################################################
# Backbone archs that are currently implemented in model/__init__.py
# 'resnet50' 'resnet101' 'mobilenetv2' 'effnetb0' up to 'effnetb7' 'densenet121' 'xception'
config.MODEL.backbone_arch = "resnet50" # 'resnet50' 'mobilenetv2' 'effnetb0'
config.MODEL.base_name = 'resnet50' #'vgg16', 'resnet50' 'mobilenetv2' 'efficientnetb2'
config.MODEL.type = "constrastive_model" #'constrastive_model' #'classification' # "autoencoder"
config.MODEL.emb_size = 512
config.MODEL.is_train_backbone= False
config.MODEL.dropout_rate = 0.2
config.MODEL.l2_regularization = True
config.MODEL.pretrained_dir= None
config.MODEL.pretrained_h5= None
config.MODEL.backbone_pretrained = True
config.MODEL.pretrained_model_fully_trainable = False # if loading weights from a previous model where arch_trainable was True, set this to True
config.MODEL.num_classes = 4
config.MODEL.hin = 180 #75#180
config.MODEL.win = 180 #75#180
#####################################################
# DATA
#####################################################
