import numpy as np
import cv2
from PIL import Image

from Models.BasicAutoencoder import BasicAutoencoder

import matplotlib.pyplot as plt


# Notes : Sources say to not use dropout .. also that batch_norm may be an overkill, and also
# to use simple ReLU ... dunno about padding='same' as well ..
# Shouldn't the decoder be using transposeconv2D ?
# They're so fcking blurry .. which might be useful


img_width = 768
img_height = 768
input_shape = (img_width, img_height, 1)
epochs = 200
batch_size = 4


# Load the previously stored data
data = np.load("Data/OK.npy")
#print(data.shape)

# Reshape to fit the desired input
data = data.reshape(data.shape[0], img_width, img_height, 1)

# Normalize the data
train_input = data.astype('float32') / 255.0
#print(train_input.shape)

# Let's create and compile a model
basic_autoencoder = BasicAutoencoder()
basic_autoencoder.create_net(input_shape)
basic_autoencoder.compile_net()

# Train it
# Should add a plot of the loss in time
basic_autoencoder.train_net(train_input, epochs=epochs, batch_size=batch_size)

plt.plot(basic_autoencoder.history.history['loss'])
plt.title('model loss - ' + basic_autoencoder.name)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('Graphs/' + basic_autoencoder.name + '_loss.png', bbox_inches = "tight")


# To see the actual reconstructed images
for i in range (0, train_input.shape[0]):
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = basic_autoencoder.predict(train_input[i].reshape(1, img_width, img_height, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(img_width, img_height)

    # Array has normalized values - need to multiply them again otherwise we get black picture
    im = Image.fromarray(reconstructed_img * 255.0)
    im = im.convert("L")
    im.save('Reconstructed/' + basic_autoencoder.name + str(i) + '.jpg')

    #cv2.imshow("reconstructed", reconstructed_img)
    #cv2.waitKey(0)
    #cv2.imwrite('Reconstructed/' + str(i) + '.jpg', reconstructed_img)
