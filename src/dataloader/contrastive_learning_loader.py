import numpy as np
import os,sys
import PIL
import PIL.Image
import tensorflow as tf
#import tensorflow_datasets as tfds
import random

from functools import partial
import matplotlib.pyplot as plt

sys.path.append('./')
from src.utils.tf_utils import get_preprocess_func
from src.train_config import config

#tf autotune for performance utilization
AUTOTUNE = tf.data.experimental.AUTOTUNE



CLASS_NAMES =['CNV', 'DRUSEN', 'DME', 'NORMAL']

preprocessing_func = get_preprocess_func(config)

def get_label_int(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == CLASS_NAMES
    # Integer encode the label
    return tf.argmax(one_hot)

def read_and_decode_image_from_path(filename,reshape_dims=[256,256]):
    # 1- read the image file 
    img = tf.io.read_file(filename)
    # 2- convert the compressed string to a 3D unit8 tensor
    img = tf.image.decode_jpeg(img,channels=3)
    # img = tf.cast(img, tf.float32)
    # 4- resize the image to the desired reshaped dim
    img = tf.image.resize(img,reshape_dims)
    # 3- convrt 3D unit8 to floats in the [0,1] range /Normalize
    # img  = tf.image.convert_image_dtype(img,tf.float32)
    img = tf.numpy_function(func=preprocessing_func, inp=[img], Tout=tf.float32)
    
    # 5- get label
    label = get_label_int(filename)
    return img, label


def configure_tfdata_for_performance(ds,batch_size, is_train=True):
    ds = ds.cache()
    if is_train:
        ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_training_tfdata(df,batch_size=32,reshape_size=[256,256]):
    images_list = list(df.path)
    print("Number of Training Samples: {:d}.".format(len(images_list)))
    # construct tf.data graph
    list_ds = tf.data.Dataset.list_files(images_list)
    train_ds = list_ds.map(partial(read_and_decode_image_from_path, reshape_dims=reshape_size), num_parallel_calls=AUTOTUNE)
    # configure training set for performance
    train_ds = configure_tfdata_for_performance(train_ds,batch_size)
    return train_ds

def get_test_or_validation_tfdata(df,batch_size=32,reshape_size=[256,256]):
    """No uagmentation and shuffling required for test/validation set

    Args:
        df (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 32.
        reshape_size (list, optional): _description_. Defaults to [256,256].

    Returns:
        _type_: _description_
    """
    images_list = list(df.path)
    print("Number of Test/Validation Samples: {:d}.".format(len(images_list)))
    # construct tf.data graph
    list_ds = tf.data.Dataset.list_files(images_list)
    test_ds = list_ds.map(partial(read_and_decode_image_from_path, reshape_dims=reshape_size), num_parallel_calls=AUTOTUNE)
    # configure training set for performance
    test_ds = configure_tfdata_for_performance(test_ds,batch_size,is_train=False)
    return test_ds

def get_eyeball_visuzalization_set_from_tf_data(tf_data):
    # get batch of data for visualization
    vis_image,vis_label = next(iter(tf_data)) # extract 1 batch from the dataset
    vis_image = vis_image.numpy()
    vis_label = vis_label.numpy()
    return vis_image, vis_label

def _denorm(img, min_image_val, max_image_val):
    """ Denormalize image by a min and max value """
    if min_image_val < 0:
        if max_image_val > 1: # [-127,127]
            img = (img + 127) / 255.
        else: # [-1,1]
            img = (img + 1.) / 2.
    elif max_image_val > 1: # no scaling
        img /= 255.
    else: # [0,1]
        pass
    return np.clip(img, 0,1)

def debug_batch_of_data(tf_data):
    for image, label in tf_data.take(1):
        image = image.numpy()
        print("Image shape: ", image.shape)
        print("Label: ", label.numpy())
        print("max img val: ", np.max(image))
        print("min img val: ", np.min(image))



### helper functions to visualize images
def show_single_image(filename,reshape_dims=[256,256]):
    img, label  =  read_and_decode_image_from_path(filename,reshape_dims=reshape_dims)
    plt.imshow(img.numpy())
    plt.title(CLASS_NAMES[label.numpy()])


def view_batch_of_images(ds,batch_size):
    image, label= next(iter(ds)) # extract 1 batch from the dataset
    image = image.numpy()
    label = label.numpy()
    print(len(label))

    fig = plt.figure(figsize=(22, 4))
    for i in range(batch_size):
        ax =  plt.subplot(1, batch_size, i+1)
        # ax.imshow(image[i])
        # plt.imshow(_denorm(image[i], np.min(image[i]), np.max(image[i]))[...,::-1])
        img = image[i]
        rgb = img[...,::-1].copy()
        dnr = _denorm(rgb, np.min(img), np.max(img))
        plt.imshow(dnr)
        # plt.imshow(_denorm(image[i], np.min(image[i]), np.max(image[i]))[...,::-1]) # denorm-> rgb to bgr (preprocess_func)
        # ax =  plt.subplot(2, batch_size, i+batch_size+1)
        # plt.imshow(_denorm(label[i], np.min(label[i]), np.max(label[i]))[...,::-1])
        ax.set_title(f"Label: {label[i]}")
    plt.show()

#### PLOT #######################################################
def plot_samples(tf_data):
    # custom visualize (notice the visualization difference)
    image_batch, label_batch = next(iter(tf_data))
    print(image_batch.shape)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # plt.imshow(image_batch[i].numpy().astype("uint8"))
        img = image_batch[i].numpy() #.astype("uint8")
        print(np.max(img),np.min(img))
        rgb = img[...,::-1].copy()
        dnr = _denorm(rgb, np.min(img), np.max(img))
        plt.imshow(dnr[:,:,::-1]) # bgr to rgb
        label = label_batch[i]
        plt.title(CLASS_NAMES[label]) # convert one hot encode to labels
        plt.axis("off")
