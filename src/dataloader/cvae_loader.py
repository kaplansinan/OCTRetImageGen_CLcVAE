import numpy as np
import os,sys
import PIL
import PIL.Image
import tensorflow as tf
#import tensorflow_datasets as tfds
import random

from functools import partial
import matplotlib.pyplot as plt

import pandas as pd

#tf autotune for performance utilization
AUTOTUNE = tf.data.experimental.AUTOTUNE

EMBEDDING_PATH = "/data/processed/contrastive_learning/train_embeddings"


def read_and_decode_image_from_path(filename,reshape_dims=[256,256]):
    # 1- read the image file 
    img = tf.io.read_file(filename)
    # 2- convert the compressed string to a 3D unit8 tensor
    img = tf.image.decode_jpeg(img,channels=3)
    # # 3- convrt 3D unit8 to floats in the [0,1] range
    # img  = tf.image.convert_image_dtype(img,tf.float32)
    # 3- convrt 3D unit8 to floats in the [-1,1] range
    img = tf.cast(img, tf.float32) / 127.5 - 1
    # 4- resize the image to the desired reshaped dim
    img = tf.image.resize(img,reshape_dims)
    
    # embedding
    path_part_list = tf.strings.split(filename, os.path.sep)
    tmp_basename = path_part_list[-1]
    #embedding path 
    embd_path = EMBEDDING_PATH + os.sep + tmp_basename + ".npy"
#     aug_img = tf.numpy_function(func=get_embedding_vector, inp=[img, img_size], Tout=tf.float32)
    emb = tf.numpy_function(numpy_map_func, inp=[embd_path], Tout=tf.float16)
    return img,emb

def configure_tfdata_for_performance(ds,batch_size, is_train=True):
    ds = ds.cache()
    if is_train:
        ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def _normalize_np_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Load the numpy files
def numpy_map_func(feature_path):
    feature = np.load(feature_path).astype(np.float16)
    feature = _normalize_np_data(feature)
    return feature #feature.reshape(-1,128)

def get_training_tfdata(df,batch_size=8,reshape_size=[256,256]):
    images_list = list(df.path)
    print("Number of Training Samples: {:d}.".format(len(images_list)))
    # construct tf.data graph
    list_ds = tf.data.Dataset.list_files(images_list)
    train_ds = list_ds.map(partial(read_and_decode_image_from_path, reshape_dims=reshape_size), num_parallel_calls=AUTOTUNE)
    # configure training set for performance
    train_ds = configure_tfdata_for_performance(train_ds,batch_size)
    return train_ds

def debug_batch_of_data(tf_data):
    for image,label in tf_data.take(1):
        image = image.numpy()
        label = label.numpy()
        print("Image shape: ", image.shape)
        print("max img val: ", np.max(image))
        print("min img val: ", np.min(image))
        print("label shape: ", label.shape)
        print("max label val: ", np.max(label))
        print("min label val: ", np.min(label))
