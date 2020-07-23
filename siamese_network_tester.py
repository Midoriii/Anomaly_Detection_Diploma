import numpy as np
import cv2
import sys, getopt
import matplotlib.pyplot as plt

from PIL import Image
from Models.BasicSiameseNet import BasicSiameseNet


img_width = 768
img_height = 768
input_shape = (img_width, img_height, 1)
# Default params
epochs = 20
batch_size = 4

# Loading data and labels
left_data = np.load('DataHuge/Pairs_Left.npy')
right_data = np.load('DataHuge/Pairs_Right.npy')
labels = np.load('DataHuge/Pairs_Labels.npy')
# Normalize the data
left_data = left_data.astype('float32') / 255.0
right_data = right_data.astype('float32') / 255.0


model = BasicSiameseNet()
# Create and compile the model
model.create_net(input_shape)
model.compile_net()
#Train it
model.train_net(left_data, right_data, labels, epochs=epochs, batch_size=batch_size)

plt.plot(model.history.history['loss'])
plt.title('model loss - ' + model.name)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('Graphs/Losses/' + model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_loss.png', bbox_inches = "tight")
plt.close('all')

plt.plot(model.history.history['accuracy'])
plt.title('model accuracy - ' + model.name)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('Graphs/Accuracies/' + model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_acc.png', bbox_inches = "tight")
plt.close('all')

prediction = model.predict(right_data[24].reshape(1, img_width, img_height, 1))
print(prediction)
