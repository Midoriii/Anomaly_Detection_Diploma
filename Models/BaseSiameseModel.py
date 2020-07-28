import numpy as np

from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import History

'''
Base Model for all Siamese Networks

@Author: Stepan Benes
'''

class BaseSiameseModel:

    def __init__(self):
        # Embedding model is a single branch of the whole siamese network,
        # used for predictions
        self.embedding = Model()
        self.model = Model()
        self.history = History()
        self.name = "BaseSiameseModel"
        self.lr = 0.0001
        return


    def create_net(self, input_shape):
        raise NotImplementedError


    def compile_net(self):
        self.model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
        return

    def train_net(self, training_input_left, training_input_right, training_labels, epochs, batch_size):
        self.history = self.model.fit([training_input_left, training_input_right], training_labels, epochs = epochs, batch_size=batch_size)
        return

    def predict(self, predict_input, prototype):
        return self.model.predict_classes([predict_input, prototype])

    def embedding_predict(self, predict_input):
        return self.embedding.predict(predict_input)

    def save_weights(self, epoch, batch_size, type):
        self.model.save_weights('Model_Saves/Weights/' + self.name + '_' + str(type) + '_e' + str(epoch) + '_b' + str(batch_size) + '_weights.h5')
        return

    def save_model(self, epoch, batch_size, type):
        self.model.save('Model_Saves/Detailed/' + self.name + '_' + str(type) + '_e' + str(epoch) + '_b' + str(batch_size) + '_detailed')
        return

    def save_embedding_weights(self, epoch, batch_size, type):
        self.model.save_weights('Model_Saves/Weights/' + self.name + '_' + str(type) + '_embedding_e' + str(epoch) + '_b' + str(batch_size) + '_weights.h5')
        return

    def save_embedding_model(self, epoch, batch_size, type):
        self.model.save('Model_Saves/Detailed/' + self.name + '_' + str(type) + '_embedding_e' + str(epoch) + '_b' + str(batch_size) + '_detailed')
        return
