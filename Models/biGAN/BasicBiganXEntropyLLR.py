'''
Copyright (c) 2021, Štěpán Beneš


Basic bigAN net, using cross entropy as loss with lower learning rate
'''
import numpy as np
from Models.biGAN.BaseBiganModel import BaseBiganModel
from Models.Losses.custom_losses import wasserstein_loss
from Models.biGAN.weightclip_constraint import WeightClip

from keras.layers import Input, Reshape, Dense, Flatten, concatenate
from keras.layers import UpSampling2D, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from keras.models import Model
from keras.optimizers import RMSprop, Adam, SGD



class BasicBiganXEntropyLLR(BaseBiganModel):

    def __init__(self, input_shape, latent_dim=24, lr=0.00005, w_clip=0.01, batch_size=4):
        super().__init__(input_shape, latent_dim, lr, w_clip, batch_size)
        self.name = "BasicBiganXEntropyLLR"
        g_optimizer = Adam(lr=self.lr, beta_1=0.5)
        d_optimizer = SGD(lr=self.lr)
        self.disc_labels_real = np.zeros((self.batch_size, 1))
        self.genc_labels_real = np.zeros((self.batch_size, 1))
        self.genc_labels_fake = np.ones((self.batch_size, 1))
        self.disc_labels_fake = np.ones((self.batch_size, 1))

        self.d = self.build_discriminator()
        self.d.compile(optimizer=d_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.g = self.build_generator()
        self.e = self.build_encoder()
        # The Discriminator part in GE model won't be trainable - GANs take turns.
        # Since the Discrimiantor itself has been previously compiled, this won't affect it.
        self.d.trainable = False
        self.ge = self.build_ge_enc()
        self.ge.compile(optimizer=g_optimizer, loss=['binary_crossentropy', 'binary_crossentropy'])
        return


    def build_generator(self):
        z_input = Input(shape=[self.latent_dim])

        x = Dense(6*6*32)(z_input)
        x = Reshape([6, 6, 32])(x)

        # 6 -> 12
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        # 12 -> 24
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        # 24 -> 48
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        # 48 -> 96
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        # 96 -> 192
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        # 192 -> 384
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(1, (3, 3), activation='tanh', padding='same')(x)

        return Model(inputs=z_input, outputs=x)


    def build_encoder(self):
        img_input = Input(shape=[self.input_shape, self.input_shape, 1])
        # 384 -> 192
        x = Conv2D(32, (3, 3), padding='same')(img_input)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        # 192 -> 96
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        # 96 -> 48
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        # 48 -> 24
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        # 24 -> 12
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        # 12 -> 6
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Flatten()(x)

        x = Dense(256)(x)
        x = LeakyReLU(0.1)(x)
        x = Dense(self.latent_dim)(x)

        return Model(inputs=img_input, outputs=x)


    def build_discriminator(self):
        img_input = Input(shape=[self.input_shape, self.input_shape, 1])
        z_input = Input(shape=[self.latent_dim])

        # Latent
        l = Dense(256)(z_input)
        l = LeakyReLU(0.1)(l)
        l = Dense(256)(l)
        l = LeakyReLU(0.1)(l)

        # Image
        x = Conv2D(64, (3, 3), padding='same')(img_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Dropout(rate=self.dropout)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Dropout(rate=self.dropout)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Dropout(rate=self.dropout)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Dropout(rate=self.dropout)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        # Joint
        x = Flatten()(x)
        x = concatenate([x, l])

        x = Dense(256)(x)
        x = LeakyReLU(0.1)(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(inputs=[img_input, z_input], outputs=x)


    def build_ge_enc(self):
        img_input = Input(shape=[self.input_shape, self.input_shape, 1])
        z_input = Input(shape=[self.latent_dim])

        fake_imgs = self.g(z_input)
        critic_fake = self.d([fake_imgs, z_input])

        fake_z = self.e(img_input)
        critic_real = self.d([img_input, fake_z])

        return Model(inputs=[img_input, z_input], outputs=[critic_real, critic_fake])
