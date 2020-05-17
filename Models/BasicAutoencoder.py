import numpy as np

from keras.layers import Input, Reshape, Dense, Flatten
from keras.layers import Activation, Conv2D, MaxPooling2D, UpSampling2D, PReLU
from keras.initializers import Constant
from keras.models import Model
from keras.callbacks import History

'''
Basic Convolutional Autoencoder

@Author: Stepan Benes
'''

class BasicAutoencoder:

    def __init__(self):
        # Not entirely sure what encoded and decoded should be initialized as
        self.encoded = Model()
        self.decoded = Model()
        self.model = Model()
        self.history = History()
        self.name = "BasicAutoencoder"
        self.filters = 32
        return


    def create_net(self, input_shape):
        net_input = Input(shape=input_shape)
        x = Conv2D(self.filters, (3, 3), padding='same')(net_input)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        self.encoded = MaxPooling2D((2, 2), padding='same')(x)

        # And now the decoder part
        x = Conv2D(self.filters, (3, 3), padding='same')(self.encoded)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = UpSampling2D((2, 2))(x)
        self.decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.model = Model(net_input, self.decoded)
        return


    def compile_net(self):
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()
        return


    def train_net(self, training_input, epochs, batch_size):
        self.history = self.model.fit(training_input, training_input, epochs = epochs, batch_size=batch_size)
        return


    def predict(self, predict_input):
        return self.model.predict(predict_input)


    def save_weights(self, epoch, batch_size):
        self.model.save_weights('Model_Saves/Weights/' + self.name + '_e' + str(epoch) + '_b' + str(batch_size) + '_weights.h5')
        return

    def save_model(self, epoch, batch_size):
        self.model.save('Model_Saves/Detailed/' + self.name + '_e' + str(epoch) + '_b' + str(batch_size) + '_detailed')
        return
