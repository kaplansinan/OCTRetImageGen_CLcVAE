# import matplotlib.pyplot as plt
import os, json, sys 
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split


def get_preprocess_func(config):
    arch = config.MODEL.backbone_arch

    if arch == 'mobilenetv2':
        return tf.keras.applications.mobilenet.preprocess_input
    elif 'resnet' in arch:
        return tf.keras.applications.resnet.preprocess_input
    elif 'eff' in arch:
        return tf.keras.applications.efficientnet.preprocess_input
    elif 'dense' in arch:
        return tf.keras.applications.densenet.preprocess_input
    elif arch == 'xception':
        return tf.keras.applications.xception.preprocess_input
    else:
        raise Exception(f'{arch} is not yet implemented')


def plot_eyeball_set_each_epoch(epoch,model,eye_ball_set,save_dir):

    decoded_imgs = model.predict(eye_ball_set)

    n = len(eye_ball_set)
    f = plt.figure(figsize=(20, 4))
    for i in range(1,n):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(eye_ball_set[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Original')

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Reconstruction')
    # plt.show()
    f.suptitle('batch_reco_vis_'+str(epoch))
    plt.subplots_adjust(top=0.95)
    if save_dir is not None:
        save_fn = os.path.join(save_dir, 'vis_'+str(epoch)+'.png')
        plt.savefig(save_fn, bbox_inches='tight', transparent=False)





# def plot_eye_set(
#     model,
#     config,
#     epoch,
#     logs,
#     dataset_fields,
#     eye_batch, 
#     out_dir, 
#     cloud_logger=None):
#     """ Custom visualization callback func """
#     eye_imgs, _ = eye_batch
#     # call with predict to get correct batchnorm behaviour
#     preds = model.predict(eye_imgs)

#     save_fn = os.path.join(out_dir, f'vis_epoch_{epoch}.png')
#     title = str(logs)
#     fig = plot_samples_preds(
#         samples=eye_batch, preds=preds, config=config, dataset_fields=dataset_fields, 
#         save_fn=save_fn, title=title, training_mode=config.TRAIN.MODE)
#     if cloud_logger is not None:
#         if epoch % 10 == 0:
#             cloud_logger.log_image_series(
#                 field="evaluation/val_sample",
#                 path=save_fn
#             )
#     plt.close('all')



#   # custom plotting callback
#     callbacks += [tf.keras.callbacks.LambdaCallback(
#         on_epoch_end= lambda epoch, logs: plot_eye_set(
#             model=model,
#             config=config, 
#             epoch=epoch, 
#             logs=logs, 
#             dataset_fields=dataset_fields,
#             eye_batch=eye_batch,
#             out_dir=vis_dir, 
#             cloud_logger=cloud_logger)
#     )]


