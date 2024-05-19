# Configuration file for training the insect classifier.
# Based on the example by Joni Juvonen / Sinan Kaplan, train_config.py.
# MM 12.11.2020

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
config.MODEL.backbone_arch = "effnetb4" # 'resnet50' 'mobilenetv2' 'effnetb0'
config.MODEL.base_name = 'resnet50' #'vgg16', 'resnet50' 'mobilenetv2' 'efficientnetb2'
config.MODEL.type = "classification" #'embedding' #'classification' # "feature_extractor"
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
config.DATA.training_csv ="F:/XAI/data/processed/OCT2017/20210819_wholeset_train.csv" # "F:/XAI/data/processed/OCT2017/df_gradcam_region_cut.csv" # "F:/XAI/data/processed/OCT2017/filtered_train_whole_set.csv"
config.DATA.testing_csv = "F:/XAI/data/processed/OCT2017/test.csv"
config.DATA.val_csv = "F:/XAI/data/processed/OCT2017/val.csv"
config.DATA.data_path = "D:/data/XAI/OCT2017/val/"
config.DATA.label_encoder = {'Anbormal':0, 'Normal':1} # {'Not_insect':0, 'Maybe_insect':1, 'Yes_insect':2}
config.DATA.save_csv_path = "D:/data/XAI//processed/OCT2017/val_win.csv"
config.DATA.output_directory = "D:/Learning/anomaly-detection-wms/data/crops/"
config.DATA.save_path= "D:/Learning/anomaly-detection-wms/data/crops/"
config.DATA.n_cv_folds = 5
config.DATA.image_size = 180 #75#180
config.DATA.class_names = ['CNV', 'DRUSEN', 'DME', 'NORMAL']