import logging
import os
import pprint
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_errors
import tensorflow as tf
from absl import app, flags

pp = pprint.PrettyPrinter(indent=4)
os.environ['TF_KERAS'] = '1'
#set random seed number for reproducibility
random.seed(2021)
np.random.seed(2021)

# Enable GPU memory growth - avoid allocating all memory at start
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

# run from root folder
import sys

sys.path.append('.')
from configs.train_config import config
from models import get_model_func

from src.callbacks import get_callbacks
from src.datagenerator import (debug_batch, get_eyeball_set_from_tf_data,
                               get_tf_training_validation_data,
                               plot_batch_samples, view_image)
from src.logger import Logger
from src.loss import get_loss_dict
from src.vis import plot_history

# Global variables
eye_batch = None
save_dir = None

def prepare_logger():
    pass
def prepare_dataset():
    pass
def prepare_model():
    pass
def train():
    pass
def main(unused_argv):
    global config, eye_batch, save_dir
    logger = Logger(root='./output/', log_level=logging.DEBUG, name="OCT2017_Training")
    save_dir = logger.logdir
    # log files
    logger.log_file(os.path.join('src/', 'train.py'))
    logger.log_file(os.path.join('configs/', 'train_config.py'))

    if len(gpus) == 0:
        logger.log(level=logging.WARNING, msg='No GPUs detected')
    
    ### DATA ###
    train_csv_path = config.DATA.training_csv
    print(train_csv_path)
    train_df = pd.read_csv(train_csv_path)
    train_df['path'] = train_df['path'].apply(lambda x: x.replace('E:', 'F:'))


    

    ### MODEL ###
    model_func = get_model_func(config)
    model = model_func()
    #keras_model.summary()
    # load weights
    #if config.MODEL.pretrained_path is not None:
        #model.load_weights(config.MODEL.pretrained_path, by_name=True)

    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    loss_dict = get_loss_dict(config)
    model.compile(
        optimizer = optimizer,
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ['accuracy']
    )

    ### TRAIN ###

    callbacks = get_callbacks(config, logger.logdir, has_validation=False)

    # custom plotting callback
    # plot_preds_cb = tf.keras.callbacks.LambdaCallback(
    #     on_epoch_end= lambda epoch, logs: plot_eye_set(model, epoch, logs, 
    #                                                    dataset_fields, cloud_logger=cloud_logger)
    # )
    # callbacks += [plot_preds_cb]
    # if cloud_log_cb is not None: callbacks += [cloud_log_cb]
    #train_data_len = int(tf.data.experimental.cardinality(tf_data_set).numpy())
    #train_iter = train_data_len // 2
    for i in range(5):
        print("Training over fold:", i)
        tmp_save_path = "my_model_20210916_wholeset_fold_"+str(i)
        tmp_df = train_df.loc[(train_df['kfold']==i)]
        tf_data_set, _ = get_tf_training_validation_data(tmp_df,batch_size=16)
        #plot_batch_samples(tf_data_set,  max_samples=9, save_file_path=save_dir, figscale=2)
        pp.pprint(debug_batch(tf_data_set))
        history = model.fit(
            tf_data_set,
            epochs = 2,
            callbacks = callbacks,
        )

        model.save(tmp_save_path)

        del tf_data_set

    #######################
    ### SAVE MODEL AND PLOT
    #plot_history(history, logger.logdir)
    
    # history = keras_model.fit(tf_data_set,epochs=10)
    # # history = keras_model.fit(tf_data_set,epochs=10,callbacks=[checkpoint])
    model.save("my_model_20210916_filtered_v4")


if __name__ == '__main__':
    # FLAGS(sys.argv)
    app.run(main)
