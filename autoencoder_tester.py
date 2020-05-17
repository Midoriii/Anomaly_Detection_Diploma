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

# Arrays to hold reconstructed images for anoamly detection by mean squared error
reconstructed_ok_array = []
reconstructed_anomalous_array = []

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
plt.title('model loss - ' + model.name)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('Graphs/Losses/' + model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_loss.png', bbox_inches = "tight")
plt.close('all')

# Save the weights and even the whole model
model.save_weights(epochs, batch_size)
model.save_model(epochs, batch_size)


# To see the actual reconstructed images of the training data
# And also to save them for MSE anomaly detection
for i in range (0, train_input.shape[0]):
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(train_input[i].reshape(1, img_width, img_height, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(img_width, img_height)

    # Append the reconstructed image to the reconstructed_ok array
    reconstructed_ok_array.append(reconstructed_img)

    # Array has normalized values - need to multiply them again otherwise we get black picture
    im = Image.fromarray(reconstructed_img * 255.0)
    im = im.convert("L")
    im.save('Reconstructed/' + model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_' + str(i) + '.jpg')

    #cv2.imshow("reconstructed", reconstructed_img)
    #cv2.waitKey(0)
    #cv2.imwrite('Reconstructed/' + str(i) + '.jpg', reconstructed_img)

# Convert to numpy array
reconstructed_ok_array = np.array(reconstructed_ok_array)
#print(reconstructed_ok_array.shape)
#print(reconstructed_ok_array[2])

# To see the actual reconstructed images of the anomalous data
# And also to save them for MSE anomaly detection
for i in range (0, anomalous_input.shape[0]):
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(anomalous_input[i].reshape(1, img_width, img_height, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(img_width, img_height)

    # Append the reconstructed image to the reconstructed_ok array
    reconstructed_anomalous_array.append(reconstructed_img)

    # Array has normalized values - need to multiply them again otherwise we get black picture
    im = Image.fromarray(reconstructed_img * 255.0)
    im = im.convert("L")
    im.save('Reconstructed/' + model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_' + str(i) + '_anomalous.jpg')

# Convert to numpy array
reconstructed_anomalous_array = np.array(reconstructed_anomalous_array)
#print(reconstructed_anomalous_array.shape)
#print(reconstructed_anomalous_array[2])

# Array to hold MSE values
reconstructed_ok_errors = []
reconstructed_anomalous_errors = []

# Compute the reconstruction MSE for ok data
for i in range (0, train_input.shape[0]):
    # Reshape into 768x768
    original_image = train_input[i].reshape(img_width, img_height)
    # Add reconstructed image MSE to the array
    reconstructed_ok_errors.append(np.square(np.subtract(original_image, reconstructed_ok_array[i])).mean())

# Same for the anomalous data
for i in range (0, anomalous_input.shape[0]):
    # Reshape into 768x768
    original_image = anomalous_input[i].reshape(img_width, img_height)
    # Add reconstructed image MSE to the array
    reconstructed_anomalous_errors.append(np.square(np.subtract(original_image, reconstructed_anomalous_array[i])).mean())

#print(reconstructed_ok_errors)
#print(reconstructed_anomalous_errors)

# Plot the MSEs
x = range (0, len(reconstructed_ok_errors))
z = range (0 + len(reconstructed_ok_errors), len(reconstructed_anomalous_errors) + len(reconstructed_ok_errors))
plt.scatter(x, reconstructed_ok_errors, c='g', s=10, marker='o', edgecolors='black', label='OK')
plt.scatter(z, reconstructed_anomalous_errors, c='r', s=10, marker='o', edgecolors='black', label='Anomalous')
# Horizontal line at 3 times the standard deviation, typical for outlier detection
plt.axhline(y= (3 * np.std(reconstructed_ok_errors)), color='r', linestyle='-')
plt.legend(loc='upper left')
plt.ylabel('Reconstruction Error')
plt.xlabel('Index')
plt.savefig('Graphs/ReconstructionErrors/' + model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_RError.png', bbox_inches = "tight")
plt.close('all')

# Save the error arrays too, so one can see which images were problematic
reconstructed_ok_errors = np.array(reconstructed_ok_errors)
reconstructed_anomalous_errors = np.array(reconstructed_anomalous_errors)

np.save('Reconstructed/Error_Arrays/' +  model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_ROK.npy', reconstructed_ok_errors)
np.save('Reconstructed/Error_Arrays/' +  model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_RAnomalous.npy', reconstructed_anomalous_errors)
