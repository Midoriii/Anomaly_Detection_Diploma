'''
Basic Siamese Network with reduced filter number
'''
import numpy as np
import keras.backend as K

from Models.Siamese.BaseSiameseModel import BaseSiameseModel

from keras.layers import Input, Reshape, Dense, Flatten, Lambda
from keras.layers import Activation, Conv2D, MaxPooling2D, BatchNormalization, Dropout, ReLU
from keras.models import Model, Sequential
from keras.callbacks import History



class SiameseNetLF(BaseSiameseModel):

    def __init__(self):
        super().__init__()
        self.name = "SiameseNetLF"
        self.lr = 0.00001
        return

    def create_net(self, input_shape):
        left_input = Input(shape=input_shape)
        right_input = Input(shape=input_shape)

        dropout_rate = 0.5

        siamese_model_branch_sequence = [
            Conv2D(32, (3, 3), padding='same'),
            Dropout(rate=dropout_rate),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D((2, 2), padding='same'),

            Conv2D(32, (3, 3), padding='same'),
            Dropout(rate=dropout_rate),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D((2, 2), padding='same'),

            Conv2D(64, (3, 3), padding='same'),
            Dropout(rate=dropout_rate),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D((2, 2), padding='same'),

            Conv2D(64, (3, 3), padding='same'),
            Dropout(rate=dropout_rate),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D((2, 2), padding='same'),

            Flatten(),
            Dense(32, activation='sigmoid')
        ]

        branch = Sequential(siamese_model_branch_sequence)

        left_embedding = branch(left_input)
        right_embedding = branch(right_input)

        # Keep the left branch as embedding model for predictions
        self.embedding = Model(left_input, left_embedding)
        # Using custom Lambda layer to compute eucldiean distance between the outputs of both branches
        distance_euclid = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([left_embedding , right_embedding])
        similarity_output = Dense(1, activation='sigmoid')(distance_euclid)

        self.model = Model([left_input, right_input], similarity_output)
        return
