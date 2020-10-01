'''
Basic Triplet Network, with higher learning rite
'''
import numpy as np
import keras.backend as K

from Models.BaseTripletModel import BaseTripletModel

from keras.layers import Input, Reshape, Dense, Flatten
from keras.layers import Activation, Conv2D, MaxPooling2D, BatchNormalization, Dropout, ReLU
from keras.models import Model, Sequential
from keras.callbacks import History



class BasicTripletNetHLR(BaseTripletModel):

    def __init__(self):
        super().__init__()
        self.name = "BasicTripletNetHLR"
        self.lr = 0.0001
        return

    def create_net(self, input_shape):
        anchor_input = Input(shape=input_shape)
        pos_input = Input(shape=input_shape)
        neg_input = Input(shape=input_shape)

        dropout_rate = 0.5

        triplet_model_branch_sequence = [
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

            Conv2D(128, (3, 3), padding='same'),
            Dropout(rate=dropout_rate),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D((2, 2), padding='same'),

            Conv2D(128, (3, 3), padding='same'),
            Dropout(rate=dropout_rate),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D((2, 2), padding='same'),

            Flatten(),
            Dense(64, activation='sigmoid')
        ]

        branch = Sequential(triplet_model_branch_sequence)

        anchor_embedding = branch(anchor_input)
        pos_embedding = branch(pos_input)
        neg_embedding = branch(neg_input)

        input = [anchor_input, pos_input, neg_input]

        # Keep the anchor branch as embedding model for predictions
        self.embedding = Model(anchor_input, anchor_embedding)
        # Output a list containing the embeddings
        output = [anchor_embedding, pos_embedding, neg_embedding]

        self.model = Model(input, output)
        return
