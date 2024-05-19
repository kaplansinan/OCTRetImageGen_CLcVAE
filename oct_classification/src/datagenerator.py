import numpy as np
import os,sys
import PIL
import PIL.Image
import tensorflow as tf
#import tensorflow_datasets as tfds
import random

from functools import partial
import matplotlib.pyplot as plt
import math
import pathlib
import cv2
from albumentations import (
    Compose, augmentations,RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate
)

sys.path.append('./')
from src.tf_image_preprocessing import get_preprocess_func
from configs.train_config import config

print(tf.__version__)
# assert tf.__version__>'2.2.1', "tensorflow version mismatch"

#tf autotune for performance utilization
AUTOTUNE = tf.data.experimental.AUTOTUNE

CLASS_NAMES = config.DATA.class_names

def get_class_names(root_path):
    # the structure of the data set can be used to get class names
    data_dir = pathlib.Path(root_path)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
    return class_names

# model specific preprocessing function
def get_model_agnostic_preprocess_func(config):
    """ Get preprocessing function based on model type (resnet, efficientnet etc)

    Args:
        model_type ([type]): [description]

    Returns:
        [type]: [description]
    """
    #https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/preprocess_input
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/preprocess_input
    # return tf.keras.applications.resnet.preprocess_input
    return get_preprocess_func(config)
    # return tf.keras.applications.efficientnet.preprocess_input

def set_shapes(img, label, img_shape=(180,180,3),num_classes=5):
    img.set_shape(img_shape)
    # label.set_shape([]) #int
    label.set_shape([num_classes]) #one hot encoded
    return img, label


# split the data into validation and training set
def split_ds_train_val(list_ds,image_count):
    val_size = int(image_count * 0.0)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)
    # # Let's now split our dataset in train and validation.
    # train_dataset = dataset.take(round(image_count * 0.8))
    # val_dataset = dataset.skip(round(image_count * 0.8))
    # get length of each data set if necessary
    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(val_ds).numpy())
    return train_ds, val_ds

def get_label_int(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == CLASS_NAMES
    # Integer encode the label
    return tf.argmax(one_hot)

def get_label_one_hot_coded(file_path):
    # part_list = tf.strings.split(path, "/")
    # part_list = tf.strings.split(path, os.path.sep)
    part_list = tf.strings.split(file_path, os.path.sep)
    # list_classes = list(config.DATA.label_encoder.keys())
    label = part_list[-2]==CLASS_NAMES
    lbl = tf.where(label)[0]
    #lbl = label_tmp_dict[part_list[-2].numpy()]
    label_tf = tf.one_hot(lbl, depth=len(CLASS_NAMES))[0]
    return label_tf

# tf2 decode image
def decode_img(img,img_size):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [img_size[0],img_size[1]])

preprocessing_func = get_preprocess_func(config)

def process_path(file_path, img_size):
    label = get_label_one_hot_coded(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img,img_size)
    # normalize
    pp_img = tf.numpy_function(func=preprocessing_func, inp=[img], Tout=tf.float32)
    return pp_img, label

def get_albu_transformations():

    transforms = Compose([
        #  # IMPORTANT - KEEP THESE RESIZE & PAD FUNCTIONS -> 
        #     augmentations.transforms.LongestMaxSize(
        #         max_size=max(config.MODEL.win, config.MODEL.hin),
        #         interpolation=cv2.INTER_LINEAR,
        #         always_apply=True
        #     ),
        #     augmentations.transforms.PadIfNeeded(
        #         min_height=config.MODEL.hin,
        #         min_width=config.MODEL.win,
        #         border_mode=cv2.BORDER_CONSTANT,
        #         value=(0,0,0),
        #         always_apply=True
        #     ),
            # <- IMPORTANT - KEEP THESE RESIZE & PAD FUNCTIONS
                Rotate(limit=60),
                RandomBrightness(limit=0.3),
                JpegCompression(quality_lower=85, quality_upper=100, p=0.8),
                HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
                RandomContrast(limit=0.2, p=0.7),
                HorizontalFlip(),
            ])
    return transforms

# https://www.programcreek.com/python/example/120573/albumentations.Resize

ALBU_TRANSFORMS = get_albu_transformations()

def aug_fn(image, img_size):
    data = {"image":image}
    aug_data = ALBU_TRANSFORMS(**data)
    aug_img = aug_data["image"]
    # aug_img = tf.cast(aug_img/255.0, tf.float32)
    aug_img = tf.image.resize(aug_img, [img_size[0],img_size[1]])
    # aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
    return aug_img


def process_path_with_albu(file_path, img_size):
    label = get_label_one_hot_coded(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # apply augmentations
    aug_img = tf.numpy_function(func=aug_fn, inp=[img, img_size], Tout=tf.float32)
    # normalize
    pp_img = tf.numpy_function(func=preprocessing_func, inp=[aug_img], Tout=tf.float32)
    return pp_img, label


def configure_for_performance(ds,batch_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

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

def debug_batch(tf_data):
    for image, label in tf_data.take(1):
        image = image.numpy()
        print("Image shape: ", image.shape)
        print("Label: ", label.numpy())
        print("max img val: ", np.max(image))
        print("min img val: ", np.min(image))

#### GENERATE TRAIN AND VAL SET #################################
def get_tf_training_validation_data(df,batch_size=32):
    IMG_WIDTH = config.MODEL.win
    IMG_HEIGHT = config.MODEL.hin
    img_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    images_list = list(df.path)
    print("Length of training image list: {:d}.".format(len(images_list)))
    list_ds = tf.data.Dataset.list_files(images_list)
    ds_train_v3 = list_ds.map(partial(process_path_with_albu, img_size=(IMG_HEIGHT,IMG_WIDTH)), num_parallel_calls=AUTOTUNE)
    # set shapes with three parameters
    set_shapes_from_config = partial(set_shapes, img_shape=img_shape, num_classes=len(CLASS_NAMES))
    ds_alb = ds_train_v3.map(set_shapes_from_config, num_parallel_calls=AUTOTUNE)
    # ds_alb = ds_train_v3.map(set_shapes, num_parallel_calls=AUTOTUNE) # this is necessary for tf.keras models
    # ds_train_v4 = configure_for_performance(ds_alb,batch_size)
    # divide train and validation sets
    image_count = len(images_list)
    train_ds, val_ds = split_ds_train_val(ds_alb,image_count)
    # configure each set for performance
    train_ds = configure_for_performance(train_ds,batch_size)
    val_ds = configure_for_performance(val_ds,batch_size)
    return train_ds, val_ds #ds_train_v4

def get_tf_test_data(df, batch_size=4):
    IMG_WIDTH = config.MODEL.win
    IMG_HEIGHT = config.MODEL.hin
    img_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    images_list = list(df.path)
    print("Length of training image list: {:d}.".format(len(images_list)))
    list_ds = tf.data.Dataset.list_files(images_list)
    ds_test_v1 = list_ds.map(partial(process_path, img_size=(IMG_HEIGHT,IMG_WIDTH)), num_parallel_calls=AUTOTUNE)
    # set shapes with three parameters
    set_shapes_from_config = partial(set_shapes, img_shape=img_shape, num_classes=len(CLASS_NAMES))
    ds_test_v2 = ds_test_v1.map(set_shapes_from_config, num_parallel_calls=AUTOTUNE)
    test_ds = configure_for_performance(ds_test_v2,batch_size)
    return test_ds

def get_eyeball_set_from_tf_data(tf_data):
    # get batch of data
    vis_image, vis_label = next(iter(tf_data)) # extract 1 batch from the dataset
    vis_image = vis_image.numpy()
    vis_label = vis_label.numpy()
    return vis_image, vis_label


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
        plt.title(class_names[np.where(label==1)]) # convert one hot encode to labels
        plt.axis("off")

def plot_batch_samples(ds,  max_samples=9,save_file_path=None, figscale:int=2):
    """ Plot batch samples from a dataset and save to fn """
    if hasattr(ds,'element_spec'): # ad-hoc test to assess if ds is tf.dataset or not...
        # draw a batch from tf.data dataset
        images, labels = next(iter(ds))
        images = images.numpy()
        labels = labels.numpy()
    else:
        # draw a batch from ImageDataGenerator-based dataset
        images, labels = ds.next()
        # images, labels = ds.take(1)
    
    # limit the number of shown images
    n_samples = min(len(images), max_samples)
    assert n_samples > 0
    root_n = int(math.ceil(math.sqrt(n_samples)))

    f, axs = plt.subplots(root_n, root_n, figsize=(root_n*figscale, root_n*figscale))
    if root_n > 1:
        axs = axs.flat
    # find denormalization parameters
    min_image_val = min([np.array(img).min() for img in images])
    max_image_val = max([np.array(img).max() for img in images])

    for i in range(n_samples):
        img = _denorm(images[i], min_image_val, max_image_val)
        # bgr to rgb
        # rgb = img[...,::-1].copy()
        ax = axs[i] if root_n > 1 else axs
        ax.imshow(img)
        ax.set_title(labels[i])
        ax.set_axis_off()
    # hide rest of the axes
    for j in range(i, root_n*root_n):
        axs[j].set_axis_off()
    plt.show()
    if save_file_path:
        plt.savefig(os.path.join(save_file_path,'batch_sample.png'))
        plt.close()
    # plt.savefig(fn, transparent=False)
    #plt.close()

# helper function to visualize images
def view_image(ds):
    image, label = next(iter(ds)) # extract 1 batch from the dataset
    image = image.numpy()
    label = label.numpy()

    fig = plt.figure(figsize=(22, 22))
    for i in range(20):
        ax = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])
        # ax.imshow(image[i])
        ax.imshow(_denorm(image[i], np.min(image[i]), np.max(image[i]))[...,::-1]) # denorm-> rgb to bgr (preprocess_func)
        ax.set_title(f"Label: {label[i]}")