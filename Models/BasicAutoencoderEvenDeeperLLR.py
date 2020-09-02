'''
Basic Convolutional Autoencoder, especially deep, the encoding is really small
With decreased learning rate
'''
import numpy as np
from Models.BaseModel import BaseModel

from keras.layers import Input, Reshape, Dense, Flatten
from keras.layers import Activation, Conv2D, MaxPooling2D, UpSampling2D, PReLU
from keras.initializers import Constant
from keras.models import Model
from keras.callbacks import History
from keras.optimizers import Adam



class BasicAutoencoderEvenDeeperLLR(BaseModel):

    def __init__(self):
        super().__init__()
        self.name = "BasicAutoencoderEvenDeeperLLR"
        return

    # Custom optimizer learning rate to see if it improves anything
    def compile_net(self):
        opt = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=opt, loss='mse')
        self.model.summary()
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
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
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

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(self.filters, (3, 3), padding='same')(x)
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
