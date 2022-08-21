import tensorflow as tf

import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import backend as K

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
