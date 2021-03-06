'''
Copyright (c) 2021, Štěpán Beneš


Base Model for all Siamese Networks
'''
import numpy as np

from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import History

from Models.Losses.custom_losses import contrastive_loss



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


    def compile_net(self, loss_func='binary_crossentropy'):
        # Has to be done this way, otherwise throws error - unknown loss func
        if loss_func == 'contrastive_loss':
            self.model.compile(optimizer=Adam(lr=self.lr), loss=contrastive_loss,
                               metrics=['binary_accuracy'])
        else:
            self.model.compile(optimizer=Adam(lr=self.lr), loss=loss_func,
                               metrics=['binary_accuracy'])
        self.model.summary()
        return

    def train_net(self, training_input_left, training_input_right, training_labels, epochs, batch_size):
        self.history = self.model.fit([training_input_left, training_input_right], training_labels, epochs=epochs, batch_size=batch_size)
        return

    def predict(self, predict_input, prototype):
        return self.model.predict([predict_input, prototype])

    def embedding_predict(self, predict_input):
        return self.embedding.predict(predict_input)

    def save_weights(self, epoch, batch_size, image_type, extended_faulty, loss_string, low_dims):
        self.model.save_weights('Model_Saves/Weights/' + str(low_dims) + self.name + '_' + str(image_type) + str(extended_faulty) + str(loss_string) + '_e' + str(epoch) + '_b' + str(batch_size) + '_weights.h5')
        return

    def save_model(self, epoch, batch_size, image_type, extended_faulty, loss_string, low_dims):
        self.model.save('Model_Saves/Detailed/' + str(low_dims) + self.name + '_' + str(image_type) + str(extended_faulty) + str(loss_string) + '_e' + str(epoch) + '_b' + str(batch_size) + '_detailed')
        return

    def save_embedding_weights(self, epoch, batch_size, image_type, extended_faulty, loss_string, low_dims):
        self.embedding.save_weights('Model_Saves/Weights/embedding_' + str(low_dims) + self.name + '_' + str(image_type) + str(extended_faulty) + str(loss_string) + '_e' + str(epoch) + '_b' + str(batch_size) + '_weights.h5')
        return

    def save_embedding_model(self, epoch, batch_size, image_type, extended_faulty, loss_string, low_dims):
        self.embedding.save('Model_Saves/Detailed/embedding_' + str(low_dims) + self.name + '_' + str(image_type) + str(extended_faulty) + str(loss_string) + '_e' + str(epoch) + '_b' + str(batch_size) + '_detailed')
        return
