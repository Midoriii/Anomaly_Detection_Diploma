'''
Base Model for all biGANs
'''
import numpy as np

from keras.models import Model

# These will be needed in actual models
from keras.optimizers import RMSprop
from Models.Losses.custom_losses import wasserstein_loss
from Models.biGAN.weightclip_constraint import WeightClip



class BaseBiganModel:

    def __init__(self, input_shape, latent_dim=24, lr=0.0001):
        self.g = Model()
        self.e = Model()
        self.d = Model()
        # This is a combined g + e model for training purposes
        self.ge = Model()

        self.d_losses = []
        self.ge_losses = []

        self.name = "BaseBiganModel"
        self.lr = lr
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.dropout = 0.2
        self.batch_size = 4

        self.d = build_discriminator()
        self.g = build_generator()
        self.e = build_encoder()
        # The Discriminator part in GE model won't be trainable - GANs take turns.
        # Sincw the Discrimiantor itself has been previously compiled, this won't affect it.
        self.d.trainable = false
        self.ge = build_ge_enc()
        return


    def build_generator(self):
        raise NotImplementedError

    def build_encoder(self):
        raise NotImplementedError

    def build_discriminator(self):
        raise NotImplementedError

    def build_ge_enc(self):
        raise NotImplementedError


    def train():
        #TODO
        return

    def predict(self, predict_input):
        #TODO - rec Error + Discrim error
        return

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
