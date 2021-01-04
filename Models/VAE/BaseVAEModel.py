'''
Copyright (c) 2021, Štěpán Beneš


Base Model for all Variational Autoencoders
'''
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.callbacks import History


class BaseVAEModel:

    def __init__(self, input_shape, latent_dim=12, lr=0.0005):
        self.e = Model()
        self.d = Model()
        self.vae = Model()
        self.encoder_mu = Model()
        self.encoder_log_variance = Model()
        self.history = History()

        self.rl_factor = 1000

        self.name = "BaseVAEModel"
        self.filters = 64
        self.lr = lr
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.dropout = 0.2
        return

    def train_net(self, training_input, epochs, batch_size):
        self.history = self.vae.fit(training_input, training_input, epochs=epochs, batch_size=batch_size)
        return

    def sampling(self, mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + K.exp(log_variance/2) * epsilon
        return random_sample

    def predict(self, orig_img):
        orig_img = orig_img.reshape(self.input_shape, self.input_shape)
        # Reshape into valid input
        img = orig_img.reshape(1, self.input_shape, self.input_shape, 1)
        # Get the reconstruction
        re_img = self.vae.predict(img).reshape(self.input_shape, self.input_shape)
        # Calculate reco error
        reconstruction_error = np.square(np.subtract(orig_img, re_img)).mean()
        return reconstruction_error

    def save_weights(self, epoch, image_type, low_dims):
        self.vae.save_weights('Model_Saves/Weights/vae_' + str(low_dims) + self.name + str(image_type) + '_e' + str(epoch) + '_weights.h5')
        return

    def save_model(self, epoch, image_type, low_dims):
        self.vae.save('Model_Saves/Detailed/vae_' + str(low_dims) + self.name + str(image_type) + '_e' + str(epoch) + '_detailed')
        return
