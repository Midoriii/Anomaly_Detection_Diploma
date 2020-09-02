'''
Base Model for all Convolutional Autoencoders
'''
import numpy as np

from keras.layers import Input, Reshape, Dense, Flatten
from keras.layers import Activation, Conv2D, MaxPooling2D, UpSampling2D, PReLU
from keras.initializers import Constant
from keras.models import Model
from keras.callbacks import History



class BaseModel:

    def __init__(self):
        # Not entirely sure what encoded and decoded should be initialized as
        self.encoded = Model()
        self.decoded = Model()
        self.model = Model()
        self.history = History()
        self.name = "BaseModel"
        self.filters = 32
        return


    def create_net(self, input_shape):
        raise NotImplementedError


    def compile_net(self):
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()
        return


    def train_net(self, training_input, epochs, batch_size):
        self.history = self.model.fit(training_input, training_input, epochs = epochs, batch_size=batch_size)
        return


    def predict(self, predict_input):
        return self.model.predict(predict_input)


    def save_weights(self, epoch, batch_size, is_data_filtered, faulty_extended):
        self.model.save_weights('Model_Saves/Weights/' + is_data_filtered + faulty_extended + self.name + '_e' + str(epoch) + '_b' + str(batch_size) + '_weights.h5')
        return

    def save_model(self, epoch, batch_size, is_data_filtered, faulty_extended):
        self.model.save('Model_Saves/Detailed/' + is_data_filtered + faulty_extended + self.name + '_e' + str(epoch) + '_b' + str(batch_size) + '_detailed')
        return

    def save_encoder_weights(self, epoch, batch_size, is_data_filtered, faulty_extended):
        self.encoded.save_weights('Model_Saves/Weights/encoder_' + is_data_filtered + faulty_extended + self.name + '_e' + str(epoch) + '_b' + str(batch_size) + '_weights.h5')
        return

    def save_encoder_model(self, epoch, batch_size, is_data_filtered, faulty_extended):
        self.encoded.save('Model_Saves/Detailed/encoder_' + is_data_filtered + faulty_extended + self.name + '_e' + str(epoch) + '_b' + str(batch_size) + '_detailed')
        return
