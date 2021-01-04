'''
Copyright (c) 2021, Štěpán Beneš


Convolutional Autoencoder with learnable Conv2DTranspose layers
'''
import numpy as np
from Models.Autoencoders.BaseModel import BaseModel

from keras.layers import Input, Reshape, Dense, Flatten
from keras.layers import Activation, Conv2D, MaxPooling2D, Conv2DTranspose, PReLU
from keras.initializers import Constant
from keras.models import Model
from keras.callbacks import History



class TransposeConvAutoencoder(BaseModel):

    def __init__(self):
        super().__init__()
        self.name = "TransposeConvAutoencoder"
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

        # Keep the encoder part
        self.encoder = Model(net_input, self.encoded)

        # And now the decoder part
        x = Conv2DTranspose(self.filters, (3,3), strides=(2,2), padding='same')(self.encoded)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        x = Conv2DTranspose(self.filters, (3,3), strides=(2,2), padding='same')(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        x = Conv2DTranspose(self.filters, (3,3), strides=(2,2), padding='same')(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        self.decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.model = Model(net_input, self.decoded)
        return
