import sys, os
import numpy as np
import pandas as pd
import os
import glob 
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import cv2
from IPython import display
# import imageio
import PIL
import time
import tensorflow as tf

import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse, binary_crossentropy
from datetime import datetime
from tqdm import tqdm

# # Enable GPU memory growth - avoid allocating all memory at start
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

import sys
sys.path.append('../')
from src.dataloader.cvae_loader import debug_batch_of_data, get_training_tfdata

def downsample(
    channels,
    kernels,
    strides=2,
    apply_norm=True,
    apply_activation=True,
    apply_dropout=False,
):
    block = keras.Sequential()
    block.add(
        layers.Conv2D(
            channels,
            kernels,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=keras.initializers.GlorotNormal(),
        )
    )
    if apply_norm:
        block.add(tfa.layers.InstanceNormalization())
    if apply_activation:
        block.add(layers.LeakyReLU(0.2))
    if apply_dropout:
        block.add(layers.Dropout(0.5))
    return block

def build_encoder(image_shape,label_shape=128, encoder_downsample_factor=64, latent_dim=128, ):
    input_image = keras.Input(shape=image_shape,name='encoder_input')
    y_labels = keras.Input(shape=label_shape, name='class_label')
    x = layers.Dense(image_shape[0] * image_shape[0]*3)(y_labels)
    x = layers.Reshape(image_shape)(x)
    x = tf.keras.layers.Concatenate()([input_image, x])
    x = downsample(encoder_downsample_factor, 3, apply_norm=False)(x)
    x = downsample(2 * encoder_downsample_factor, 3)(x)
    x = downsample(4 * encoder_downsample_factor, 3)(x)
    x = downsample(8 * encoder_downsample_factor, 3)(x)
    x = downsample(8 * encoder_downsample_factor, 3)(x)
    x = downsample(8 * encoder_downsample_factor, 3)(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_variance = layers.Dense(latent_dim, name="z_variance")(x)
#     z = layers.Lambda(sampling,
#            output_shape=(latent_dim,),
#            name='z')([z_mean, z_variance])
    encoder = keras.Model([input_image, y_labels],
                [z_mean, z_variance], 
                name='encoder')
    return encoder

def build_decoder(emb_shape=128, latent_dim=128, upscale_factor=4):
    latent = keras.Input(shape=(latent_dim),name='latent_vect')
    y_label = keras.Input(shape=emb_shape,name='class_label')
    x = tf.keras.layers.Concatenate()([latent, y_label])
    x = layers.Dense(16384)(x)
    x = layers.Reshape((4, 4, 1024))(x)
    x = layers.Conv2DTranspose(
                filters=256, kernel_size=3, strides=2, padding='same',
                activation='relu')(x) #8
    x = layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=2, padding='same',
                activation='relu')(x) #16
    x = layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu')(x) #32 
    x = layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu')(x) #64
    
    x = layers.Conv2D(64, 5, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    # x = layers.Conv2D(3 * (upscale_factor ** 2), 3, padding='same', activation='relu')(x) # no way
    x = layers.Conv2D(3 * (upscale_factor ** 2), 3, padding='same', activation='tanh')(x)
    output_image = tf.nn.depth_to_space(x, upscale_factor)
    return keras.Model([latent, y_label], output_image, name="decoder")


class CCVAE(tf.keras.Model):
    """Conditional Convolutional variational autoencoder."""

    def __init__(self, image_shape,cond_label_shape=128,encoder_downsample_factor=64,latent_dim=128):
        super(CCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.cond_label_shape= cond_label_shape
        self.encoder_donwsample_factor = encoder_downsample_factor
        #encoder 
        self.encoder = build_encoder(self.image_shape,label_shape=self.cond_label_shape, 
                                     encoder_downsample_factor=self.encoder_donwsample_factor, latent_dim=self.latent_dim)
        #decoder 
        self.decoder =  build_decoder(emb_shape=self.cond_label_shape, latent_dim=self.latent_dim)


    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar


    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

class FeatureMatchingLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mae = keras.losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        loss = 0
        for i in range(len(y_true) - 1):
            loss += self.mae(y_true[i], y_pred[i])
        return loss


class VGGFeatureMatchingLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
        layer_outputs = [vgg.get_layer(x).output for x in self.encoder_layers]
        self.vgg_model = keras.Model(vgg.input, layer_outputs, name="VGG")
        self.mae = keras.losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        y_true = keras.applications.vgg19.preprocess_input(127.5 * (y_true + 1))
        y_pred = keras.applications.vgg19.preprocess_input(127.5 * (y_pred + 1))
        real_features = self.vgg_model(y_true)
        fake_features = self.vgg_model(y_pred)
        loss = 0
        for i in range(len(real_features)):
            loss += self.weights[i] * self.mae(real_features[i], fake_features[i])
        return loss

def kl_divergence_loss(mean, variance):
    return -0.5 * tf.reduce_sum(1 + variance - tf.square(mean) - tf.exp(variance))

MSE = tf.keras.losses.MeanSquaredError()
def reconstruction_loss(y_true, y_pred):
#     mse = tf.keras.losses.MeanSquaredError()
    return MSE(y_true,y_pred)

vgg_loss = VGGFeatureMatchingLoss()
HINGE_LOSS = tf.keras.losses.Hinge()

feature_matching_loss = FeatureMatchingLoss()

def vae_loss(inputs, outputs,z_mean,z_log_var, image_size=256):
    beta = 1.0
    # reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    # reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5 * beta
    tmp_vgg_loss = vgg_loss(inputs, outputs)
    tmp_feat_loss = feature_matching_loss(inputs, outputs)
    # tmp_hng_loss = HINGE_LOSS(inputs, outputs)
    # cvae_loss = K.mean(reconstruction_loss + kl_loss + tmp_vgg_loss)
    # cvae_loss = K.mean(reconstruction_loss + kl_loss)
    # cvae_loss = K.mean(tmp_feat_loss+kl_loss)
    # cvae_loss = K.mean(tmp_feat_loss+reconstruction_loss + kl_loss)
    # cvae_loss = K.mean(5*tmp_vgg_loss+2*tmp_feat_loss+reconstruction_loss + kl_loss)
    # cvae_loss = K.mean(tmp_hng_loss+reconstruction_loss + kl_loss)
    # cvae_loss = K.mean(tmp_hng_loss+reconstruction_loss + kl_loss + tmp_feat_loss)
    cvae_loss = K.mean(25*tmp_vgg_loss + 10*tmp_feat_loss+0.1*kl_loss) # best result so far
    # cvae_loss = K.mean(tmp_vgg_loss + kl_loss)
    return cvae_loss

optimizer = tf.keras.optimizers.Adam(1e-3)
# optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
          -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=raxis)

def compute_loss_v3(model, x_image, x_cond_labels):
        mean, logvar = model.encode([x_image, x_cond_labels])
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode([z, x_cond_labels])
        return vae_loss(x_image, x_logit,mean,logvar,image_size=256)

@tf.function
def train_step(model, x_image, x_cond_labels, optimizer):
    """Executes one training step and returns the loss.

      This function computes the loss and gradients, and uses the latter to
      update the model's parameters.
      """
    with tf.GradientTape() as tape:
#         loss = compute_loss(model, x_image, x_cond_labels)
        loss = compute_loss_v3(model, x_image, x_cond_labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

epochs = 200
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 128
image_shape = (256,256,3)
num_examples_to_generate = 8
batch_size = 4

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CCVAE(image_shape)


train_csv_path =  "E:/XAI/data/processed/OCT2017/filtered_train_whole_set_pred.csv" #"F:/XAI/data/processed/OCT2017/filtered_train_whole_set.csv"
print(train_csv_path)
train_df = pd.read_csv(train_csv_path)
# PATH CONVERSION IF NEEDED
train_df['path'] = train_df['path'].apply(lambda x: x.replace('F:', 'E:')).reset_index(drop=True)
# train_df.head()
# train_df = train_df.loc[1:100]
# get data paths - exclude healthy cases
train_df = train_df[train_df['label']!='NORMAL']
tf_train_set = get_training_tfdata(train_df,batch_size=4)

# https://keras.io/examples/vision/super_resolution_sub_pixel/#build-a-model
LOG_DIR = os.path.join('../output',datetime.now().strftime("%Y%m%d-%H%M%S")+'_cvae_subpixel_kl_reco_feat_loss')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def generate_and_save_images(model, epoch, test_sample,apply_sigmo=False):
    x_images_test, x_cond_labels_test = test_sample
    mean, logvar = model.encode([x_images_test, x_cond_labels_test])
    z = model.reparameterize(mean, logvar)
#     print(z.shape)
#     print(z)
#     print(mean,logvar)
    predictions = model.decode([z, x_cond_labels_test],apply_sigmoid=apply_sigmo)
    
#     predictions = model.sample(z)
    fig = plt.figure(figsize=(20, 20))

    for i in range(predictions.shape[0]):
        ax = plt.subplot(4, 4, i*2 + 1)
        # plt.imshow(x_images_test[i, :, :, :])
        plt.imshow((x_images_test[i, :, :, :] + 1) / 2)
        ax.set_title("Real")
        plt.axis('off')
        ax = plt.subplot(4, 4, i*2 + 2)
        # plt.imshow(predictions[i, :, :, :])
        plt.imshow((predictions[i, :, :, :] + 1) / 2)
        ax.set_title("Predictions")
        plt.axis('off')

      # tight_layout minimizes the overlap between 2 sub-plots
    if apply_sigmo:
        plt.savefig(LOG_DIR +'/image_at_epoch_{:04d}_sigmoid.png'.format(epoch))
    else:
        plt.savefig(LOG_DIR +'/image_at_epoch_{:04d}_nosigmoid.png'.format(epoch))
    # plt.show()
    plt.cla()
    plt.close(fig)

for test_batch in tf_train_set.take(1):
    test_sample = test_batch

for epoch in range(0, epochs + 1):
    start_time = time.time()
    for train_x in tqdm(tf_train_set,"running training"):
#         train_step(model, train_x, optimizer)
        tmp_x_images, tmp_x_labels = train_x
        train_step(model, tmp_x_images, tmp_x_labels, optimizer)
    end_time = time.time()

    display.clear_output(wait=False)
#     print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
#         .format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, test_sample)
    #save model
    if epoch % 20==0:
        filepath_encoder=os.path.join(LOG_DIR,"model_encoder_"+str(epoch)+".h5")
        filepath_decoder=os.path.join(LOG_DIR,"model_decoder_"+str(epoch)+".h5")
        model.encoder.save_weights(filepath_encoder)
        model.decoder.save_weights(filepath_decoder)