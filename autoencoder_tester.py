import numpy as np
import cv2
from PIL import Image
import sys, getopt

from Models.BasicAutoencoder import BasicAutoencoder

import matplotlib.pyplot as plt


# Notes : Sources say to not use dropout .. also that batch_norm may be an overkill, and also
# to use simple ReLU ... dunno about padding='same' as well ..
# Shouldn't the decoder be using transposeconv2D ?
# They're so fcking blurry .. which might be useful


img_width = 768
img_height = 768
input_shape = (img_width, img_height, 1)

# Default params
epochs = 2
batch_size = 1
desired_model = "BasicAutoencoder"

# Arrays to hold reconstructed images for anoamly detection by mean squared error
reconstructed_ok_array = np.array([])
reconstructed_anomalous_array = np.array([])

# Get full command-line arguments
full_cmd_arguments = sys.argv
# Keep all but the first
argument_list = full_cmd_arguments[1:]
# Getopt options
short_options = "e:b:m:"
long_options = ["epochs=", "batch_size=", "model="]
# Get the arguments and their respective values
arguments, values = getopt.getopt(argument_list, short_options, long_options)

# Evaluate given options
for current_argument, current_value in arguments:
    if current_argument in ("-e", "--epochs"):
        epochs = int(current_value)
        #print (epochs)
    elif current_argument in ("-b", "--batch_size"):
        batch_size = int(current_value)
        #print (batch_size)
    elif current_argument in ("-m", "--model"):
        desired_model = current_value
        #print (desired_model)


# Load the previously stored data
part1 = np.load("Data/OK_1.npy")
part2 = np.load("Data/OK_2.npy")
data = np.concatenate((part1, part2))
print(data.shape)
# Load the anomalies too
anomalies = np.load("Data/Vadne.npy")
print(anomalies.shape)

# Reshape to fit the desired input
data = data.reshape(data.shape[0], img_width, img_height, 1)
anomalous_data = anomalies.reshape(anomalies.shape[0], img_width, img_height, 1)

# Normalize the data
train_input = data.astype('float32') / 255.0
anomalous_input = anomalous_data.astype('float32') / 255.0
#print(train_input.shape)

# Choose the correct model
if desired_model == "BasicAutoencoder":
    model = BasicAutoencoder()
else:
    print("No model specified")
    sys.exit()


# Create and compile the model
model.create_net(input_shape)
model.compile_net()

# Train it
# Should add a plot of the loss in time
model.train_net(train_input, epochs=epochs, batch_size=batch_size)

plt.plot(model.history.history['loss'])
plt.title('model loss - ' + nodel.name)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('Graphs/' + model.name + '_loss.png', bbox_inches = "tight")

# Save the weights and even the whole model
model.save_weights()
model.save_model()


# To see the actual reconstructed images of the training data
# And also to save them for MSE anomaly detection
for i in range (0, train_input.shape[0]):
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(train_input[i].reshape(1, img_width, img_height, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(img_width, img_height)

    # Append the reconstructed image to the reconstructed_ok array
    numpy.append(reconstructed_ok_array, reconstructed_img)

    # Array has normalized values - need to multiply them again otherwise we get black picture
    im = Image.fromarray(reconstructed_img * 255.0)
    im = im.convert("L")
    im.save('Reconstructed/' + model.name + str(i) + '.jpg')

    #cv2.imshow("reconstructed", reconstructed_img)
    #cv2.waitKey(0)
    #cv2.imwrite('Reconstructed/' + str(i) + '.jpg', reconstructed_img)


# To see the actual reconstructed images of the anomalous data
# And also to save them for MSE anomaly detection
for i in range (0, anomalous_input.shape[0]):
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(anomalous[i].reshape(1, img_width, img_height, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(img_width, img_height)

    # Append the reconstructed image to the reconstructed_ok array
    numpy.append(reconstructed_anomalous_array, reconstructed_img)

    # Array has normalized values - need to multiply them again otherwise we get black picture
    im = Image.fromarray(reconstructed_img * 255.0)
    im = im.convert("L")
    im.save('Reconstructed/' + model.name + str(i) + '_anomalous.jpg')


print(reconstructed_ok_array.shape)
print(reconstructed_ok_array[2])
print(reconstructed_anomalous_array.shape)
print(reconstructed_anomalous_array[2])
