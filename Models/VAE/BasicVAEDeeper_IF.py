'''
Copyright (c) 2021, Štěpán Beneš


Basic Variational Autoencoder, slightly deeper and with increasing filter size
'''
import numpy as np
from keras import backend as K
from Models.VAE.BaseVAEModel import BaseVAEModel

from keras.layers import Input, Reshape, Dense, Flatten, BatchNormalization, Lambda
from keras.layers import Activation, Conv2D, MaxPooling2D, UpSampling2D, PReLU
from keras.initializers import Constant
from keras.models import Model
from keras.callbacks import History
from keras.optimizers import Adam
from Models.Losses.custom_losses import vae_loss_func



class BasicVAEDeeper_IF(BaseVAEModel):

    def __init__(self, input_shape, latent_dim=12, lr=0.0005):
        super().__init__(input_shape, latent_dim, lr)

        self.name = "BasicVAEDeeper_IF"

        self.optimizer = Adam(lr=self.lr)
        self.filters = 64

        self.create_net()

        self.vae.summary()
        self.vae.compile(self.optimizer, loss=vae_loss_func(self.encoder_mu,
                                                            self.encoder_log_variance,
                                                            self.rl_factor))
        return


    def create_net(self):
        e_input = Input(shape=(self.input_shape, self.input_shape, 1))
        x = Conv2D(self.filters, (3, 3), padding='same')(e_input)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.filters*2, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.filters*2, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.filters*2, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.filters*3, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.filters*3, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        shape_before_flatten = K.int_shape(x)[1:]
        x = Flatten()(x)

        self.encoder_mu = Dense(self.latent_dim)(x)
        self.encoder_log_variance = Dense(self.latent_dim)(x)

        e_output = Lambda(self.sampling)([self.encoder_mu, self.encoder_log_variance])

        # Keep the encoder part
        self.e = Model(e_input, e_output)

        # And now the decoder part
        d_input = Input(shape=[self.latent_dim])
        x = Dense(np.prod(shape_before_flatten))(d_input)
        x = Reshape(target_shape=shape_before_flatten)(x)

        x = Conv2D(self.filters*3, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(self.filters*3, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(self.filters*2, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(self.filters*2, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(self.filters*2, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = UpSampling2D((2, 2))(x)
        d_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.d = Model(d_input, d_output)

        # Finalize the VAE
        vae_input = Input(shape=(self.input_shape, self.input_shape, 1))
        vae_enc_out = self.e(vae_input)
        vae_dec_out = self.d(vae_enc_out)

        self.vae = Model(vae_input, vae_dec_out)
        return
