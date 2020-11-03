'''
Base Model for all biGANs
'''
import numpy as np

from keras.models import Model



class BaseBiganModel:

    def __init__(self, input_shape, latent_dim=24, lr=0.0001, w_clip=0.01):
        self.g = Model()
        self.e = Model()
        self.d = Model()
        # This is a combined g + e model for training purposes
        self.ge = Model()

        self.d_losses = []
        self.ge_losses = []

        self.name = "BaseBiganModel"
        self.lr = lr
        self.w_clip = w_clip
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.dropout = 0.2
        self.batch_size = 4

        self.labels_real = np.ones((self.batch_size, 1))
        self.labels_fake = -np.ones((self.batch_size, 1))
        return


    def build_generator(self):
        raise NotImplementedError

    def build_encoder(self):
        raise NotImplementedError

    def build_discriminator(self):
        raise NotImplementedError

    def build_ge_enc(self):
        raise NotImplementedError


    def train(self, images, epochs):
        for epoch in range(epochs):
            # D training
            noise = self.latent_noise(self.batch_size, self.latent_dim)
            img_batch = self.get_image_batch(images, self.batch_size)

            fake_noise = self.e.predict(img_batch)
            fake_img_batch = self.g.predict(noise)

            d_real_loss = self.d.train_on_batch([img_batch, fake_noise], self.labels_real)
            d_fake_loss = self.d.train_on_batch([fake_img_batch, noise], self.labels_fake)
            d_loss = (0.5 * np.add(d_real_loss, d_fake_loss))
            self.d_losses.append(d_loss)
            # E+G training
            ge_enc_loss = self.ge.train_on_batch([img_batch, noise],
                                                 [self.labels_real, self.labels_fake])
            self.ge_losses.append(ge_enc_loss)

            print("Epoch: " + str(epoch) + ", D loss: " + str(d_loss[0])
                  + "; D acc: " + str(d_loss[1]) + "; E loss: " + str(ge_enc_loss[0])
                  + "; G loss: " + str(ge_enc_loss[1]))
        return

    def predict(self, orig_img):
        orig_img = orig_img.reshape(self.input_shape, self.input_shape)
        # Reshape into valid input
        img = orig_img.reshape(1, self.input_shape, self.input_shape, 1)
        # Get the latent representation of the input image
        z = self.e.predict(img)
        # Get the reconstruction
        re_img = self.g.predict(z).reshape(self.input_shape, self.input_shape)
        # Calculate both errors
        reconstruction_error = np.square(np.subtract(orig_img, re_img)).mean()
        critic_error = self.d.predict([img, z])
        return reconstruction_error + critic_error

    def latent_noise(batch_size, latent_dim):
        return np.random.normal(0.0, 1.0, size=(batch_size, latent_dim))

    def get_image_batch(images, batch_size):
        idx = np.random.randint(0, images.shape[0], batch_size)
        return images[idx]

    def save_weights(self, epoch, image_type, low_dims):
        self.g.save_weights('Model_Saves/Weights/' + str(low_dims) + self.name + '_Generator_' + str(image_type) + '_e' + str(epoch) + '_weights.h5')
        self.e.save_weights('Model_Saves/Weights/' + str(low_dims) + self.name + '_Encoder_' + str(image_type) + '_e' + str(epoch) + '_weights.h5')
        self.d.save_weights('Model_Saves/Weights/' + str(low_dims) + self.name + '_Discriminator_' + str(image_type) + '_e' + str(epoch) + '_weights.h5')
        return

    def save_model(self, epoch, image_type, low_dims):
        self.g.save('Model_Saves/Detailed/' + str(low_dims) + self.name + '_Generator_' + str(image_type) + '_e' + str(epoch) + '_detailed')
        self.e.save('Model_Saves/Detailed/' + str(low_dims) + self.name + '_Encoder_' + str(image_type) + '_e' + str(epoch) + '_detailed')
        self.d.save('Model_Saves/Detailed/' + str(low_dims) + self.name + '_Discriminator_' + str(image_type) + '_e' + str(epoch) + '_detailed')
        return
