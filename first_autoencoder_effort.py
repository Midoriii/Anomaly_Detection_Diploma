import numpy as np
import cv2

from keras.layers import Input, Reshape, Dense, Flatten
from keras.layers import Activation, Conv2D, MaxPooling2D, UpSampling2D, PReLU
from keras.initializers import Constant
from keras.models import Model

import matplotlib.pyplot as plt



# Notes : Sources say to not use dropout .. also that batch_norm may be an overkill, and also
# to use simple ReLU ... dunno about padding='same' as well ..
# Shouldn't the decoder be using transposeconv2D ?
# They're so fcking blurry


img_width = 768
img_height = 768
input_shape = (img_width, img_height, 1)
epochs = 20
batch_size = 1


# Load the previously stored data
data = np.load("Data/OK.npy")
#print(data.shape)

# Reshape to fit the desired input
data = data.reshape(data.shape[0], img_width, img_height, 1)

# Normalize the data
train_input = data.astype('float32') / 255.0
#print(train_input.shape)

# Let's define a testing model
# First the encoder part
net_input = Input(shape=input_shape)
x = Conv2D(32, (3, 3), padding='same')(net_input)
x = PReLU(alpha_initializer=Constant(value=0.25))(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = PReLU(alpha_initializer=Constant(value=0.25))(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = PReLU(alpha_initializer=Constant(value=0.25))(x)
encoder = MaxPooling2D((2, 2), padding='same')(x)

# And now the decoder part
x = Conv2D(32, (3, 3), padding='same')(encoder)
x = PReLU(alpha_initializer=Constant(value=0.25))(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = PReLU(alpha_initializer=Constant(value=0.25))(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = PReLU(alpha_initializer=Constant(value=0.25))(x)
x = UpSampling2D((2, 2))(x)
decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# THe whole autoencoder
autoencoder = Model(net_input, decoder)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

history = autoencoder.fit(train_input, train_input, epochs=epochs, batch_size=batch_size)

testing_img = autoencoder.predict(train_input[0].reshape(1, 768, 768, 1))

testing_img = testing_img.reshape(768, 768)
cv2.imshow("reconstructed", testing_img)
cv2.waitKey(0)
