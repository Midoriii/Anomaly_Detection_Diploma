'''
Base Model for all Triplet Networks
'''
import numpy as np

from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import History

from Models.Losses.custom_losses import triplet_loss



class BaseTripletModel:

    def __init__(self):
        # Embedding model is a single branch of the whole triplet network,
        # used for predictions
        self.embedding = Model()
        self.model = Model()
        self.history = History()
        self.name = "BaseTripletModel"
        self.lr = 0.0001
        return


    def create_net(self, input_shape):
        raise NotImplementedError


    def compile_net(self):
        self.model.compile(optimizer=Adam(lr=self.lr), loss=triplet_loss)
        self.model.summary()
        return

    def train_net(self, anchor, positive, negative, epochs, batch_size):
        # A dummy labels array
        labels = np.empty((3, anchor.shape[0], 1))
        self.history = self.model.fit([anchor, positive, negative], labels, epochs=epochs, batch_size=batch_size)
        return

    def predict(self, predict_input):
        return self.embedding.predict(predict_input)

    def save_weights(self, epoch, batch_size, image_type, image_set, dimensions):
        self.model.save_weights('Model_Saves/Weights/' + str(dimensions) + self.name + '_' + str(image_type) + "_set_" + str(image_set) + '_e' + str(epoch) + '_b' + str(batch_size) + '_weights.h5')
        return

    def save_model(self, epoch, batch_size, image_type, image_set, dimensions):
        self.model.save('Model_Saves/Detailed/' + str(dimensions) + self.name + '_' + str(image_type) + "_set_" + str(image_set) + '_e' + str(epoch) + '_b' + str(batch_size) + '_detailed')
        return

    def save_embedding_weights(self, epoch, batch_size, image_type, image_set, dimensions):
        self.embedding.save_weights('Model_Saves/Weights/embedding_' + str(dimensions) + self.name + '_' + str(image_type) + "_set_" + str(image_set) + '_e' + str(epoch) + '_b' + str(batch_size) + '_weights.h5')
        return

    def save_embedding_model(self, epoch, batch_size, image_type, image_set, dimensions):
        self.embedding.save('Model_Saves/Detailed/embedding_' + str(dimensions) + self.name + '_' + str(image_type) + "_set_" + str(image_set) + '_e' + str(epoch) + '_b' + str(batch_size) + '_detailed')
        return
