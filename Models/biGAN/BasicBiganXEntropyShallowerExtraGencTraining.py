'''
Basic bigAN net, using cross entropy as loss and shallower architecture
with extra G and E training
'''
import numpy as np
from Models.biGAN.BaseBiganModel import BaseBiganModel
from Models.Losses.custom_losses import wasserstein_loss
from Models.biGAN.weightclip_constraint import WeightClip

from keras.layers import Input, Reshape, Dense, Flatten, concatenate
from keras.layers import UpSampling2D, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from keras.models import Model
from keras.optimizers import RMSprop, Adam, SGD



class BasicBiganXEntropyShallowerExtraGencTraining(BaseBiganModel):

    def __init__(self, input_shape, latent_dim=48, lr=0.0005, w_clip=0.01, batch_size=4):
        super().__init__(input_shape, latent_dim, lr, w_clip, batch_size)
        self.name = "BasicBiganXEntropyShallowerExtraGencTraining"
        g_optimizer = Adam(lr=self.lr, beta_1=0.5)
        d_optimizer = SGD(lr=self.lr)
        self.disc_labels_fake = np.zeros((self.batch_size, 1))
        self.genc_labels_fake = np.zeros((self.batch_size, 1))

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

        x = Dense(24*24*32)(z_input)
        x = Reshape([24, 24, 32])(x)

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


    def train(self, images, epochs):
        for epoch in range(epochs):
            # D training
            noise = self.latent_noise(self.batch_size, self.latent_dim)
            img_batch = self.get_image_batch(images, self.batch_size)

            fake_noise = self.e.predict(img_batch)
            fake_img_batch = self.g.predict(noise)

            d_real_loss = self.d.train_on_batch([img_batch, fake_noise], self.disc_labels_real)
            self.dr_losses.append(d_real_loss[0])
            self.dr_acc.append(d_real_loss[1])
            d_fake_loss = self.d.train_on_batch([fake_img_batch, noise], self.disc_labels_fake)
            self.df_losses.append(d_fake_loss[0])
            self.df_acc.append(d_fake_loss[1])
            d_loss = (0.5 * np.add(d_real_loss, d_fake_loss))
            self.d_losses.append(d_loss[0])
            # E+G training
            ge_enc_loss = np.empty(3)
            for _ in range(0, 10):
                noise = self.latent_noise(self.batch_size, self.latent_dim)
                img_batch = self.get_image_batch(images, self.batch_size)

                ge_enc_loss += self.ge.train_on_batch([img_batch, noise],
                                                      [self.genc_labels_fake, self.genc_labels_real])
            self.e_losses.append(ge_enc_loss[1]/5.0)
            self.g_losses.append(ge_enc_loss[2]/5.0)

            print("Epoch: " + str(epoch) + ", D loss: " + str(d_loss[0])
                  + "; D acc: " + str(d_loss[1]) + "; E loss: " + str(ge_enc_loss[1]/5.0)
                  + "; G loss: " + str(ge_enc_loss[2]/5.0))
        return
