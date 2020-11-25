'''
The purpose of this script is to test various Autoencoder models on provided
images.

The type of images to use for training and testing is given by the parameters -d,
-f, respectively, and the model to be trained and tested is given by -m.
The number of training epochs by -e and the batch size to be used by -b.

The model is trained exclusively on the OK images. After training is done, the
model's loss is plotted to a graph and saved. The model itself and the weights
are saved too, along with the model's Encoder part, which transforms input into
features.

For performance evaluation, the OK, extra OK, and Faulty images are served to the trained
model (using Model.predict()) and the reconstructions are saved to OK and Faulty
lists. Afterwards, for each image, the MSE of the difference of the original image
and its reconstruction is computed and serves as a 'score' of sorts.

The idea is that a large reconstruction error points to an anomaly, as the model
is trained only on the OK data, and should struggle to reconstruct anomalous input.

To visualize the performance a graph is plotted of images and their respective
reconstruction errors. Green dots = truly OK images, red dots = truly Faulty images.

Common practice for finding a threshold for anomaly detection is to use 3 times the
standard deviation of OK scores. Such threshold is also shown on the graph and
serves as a divider of sorts showing which images are possibly faulty and which are OK.
Of course any such threshold can be edited to better serve FP vs FN needs.

Reconstructions of all images are also saved, to gain some additional insight into
the performance of the models.


Arguments:
    -e / --epochs: Desired number of epochs for the model to train for.
    -b / --batch_size: Training batch_size to be used, can't handle more than 4
    reliably on most GPUs due to VRAM limits.
    -m / --model: Name of the model class to be instantiated and used.
    -d / --data: Type of data to be used, if none given, all of the OK images
    are used, regardless of type. If 'BSE' or 'SE' given, only such images are
    used. If 'filtered' is given only the hand-picked best OK images are used.
    If 'low_dim' is given, lower dimensions variant of data is used.
    -f / --faulty: If 'extended' given, all of the faulty images are used for
    testing, including those with plugged central hole.
'''
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from Models.Autoencoders.BasicAutoencoder import BasicAutoencoder
from Models.Autoencoders.BasicAutoencoder_LF import BasicAutoencoder_LF
from Models.Autoencoders.HighStrideAutoencoder import HighStrideAutoencoder
from Models.Autoencoders.BasicAutoencoderDeeper import BasicAutoencoderDeeper
from Models.Autoencoders.BasicAutoencoderDeeperExtraLLR import BasicAutoencoderDeeperExtraLLR
from Models.Autoencoders.HighStrideAutoencoderDeeper import HighStrideAutoencoderDeeper
from Models.Autoencoders.BasicAutoencoderEvenDeeper import BasicAutoencoderEvenDeeper
from Models.Autoencoders.BasicAutoencoderEvenDeeperLLR import BasicAutoencoderEvenDeeperLLR
from Models.Autoencoders.BasicAutoencoderEvenDeeperExtraLLR import BasicAutoencoderEvenDeeperExtraLLR
from Models.Autoencoders.BasicAutoencoderEvenDeeperExtraLLRMultipleConv import BasicAutoencoderEvenDeeperExtraLLRMultipleConv
from Models.Autoencoders.TransposeConvAutoencoder import TransposeConvAutoencoder
from Models.Autoencoders.TransposeConvAutoencoderDeepExtraLLR import TransposeConvAutoencoderDeepExtraLLR
from Models.Autoencoders.HighStrideTransposeConvAutoencoder import HighStrideTransposeConvAutoencoder
from Models.Autoencoders.BasicAutoencoderLFDeeperLLR import BasicAutoencoderLFDeeperLLR
from Models.Autoencoders.BasicAutoencoderHFDeeperLLR import BasicAutoencoderHFDeeperLLR
from Models.Autoencoders.BasicAutoencoderEvenDeeperSuperLLR import BasicAutoencoderEvenDeeperSuperLLR

# Constants
IMG_WIDTH = 768
IMG_HEIGHT = 768

# Default params
epochs = 2
batch_size = 1
desired_model = "BasicAutoencoder"
# Selection of full OK dataset or only filtered best ones
# Or filtering by image type - BSE vs SE
is_data_filtered = ""
# Selection of full Faulty dataset, with plugged central holes
faulty_extended = ""

# Get full command-line arguments
full_cmd_arguments = sys.argv
# Keep all but the first
argument_list = full_cmd_arguments[1:]
# Getopt options
short_options = "e:b:m:d:f:"
long_options = ["epochs=", "batch_size=", "model=", "data=", "faulty="]
# Get the arguments and their respective values
arguments, values = getopt.getopt(argument_list, short_options, long_options)

# Evaluate given options
for current_argument, current_value in arguments:
    if current_argument in ("-e", "--epochs"):
        epochs = int(current_value)
    elif current_argument in ("-b", "--batch_size"):
        batch_size = int(current_value)
    elif current_argument in ("-m", "--model"):
        desired_model = current_value
    elif current_argument in ("-d", "--data"):
        is_data_filtered = current_value
    elif current_argument in ("-f", "--faulty"):
        faulty_extended = current_value


# If we want to use lower dimensional data, rewrite the constants
if is_data_filtered == "low_dims":
    IMG_WIDTH = 384
    IMG_HEIGHT = 384

# Load the previously stored data
# If filtered is desired, load those
if is_data_filtered == "filtered":
    train_input = np.load("Data/OK_filtered.npy")
    test_input = np.load("Data/OK_extra.npy")
    # Add the underscore so that later graph and file
    # names contain the 'filtered_' prefix when expected
    is_data_filtered = "filtered_"
# If we want only the BSE Data
elif is_data_filtered == "BSE":
    train_input = np.load("Data/BSE_ok.npy")
    test_input = np.load("Data/BSE_ok_extra.npy")
    is_data_filtered = "BSE_"
# If we want only the SE Data
elif is_data_filtered == "SE":
    train_input = np.load("Data/SE_ok.npy")
    test_input = np.load("Data/SE_ok_extra.npy")
    is_data_filtered = "SE_"
# If we want only the lower dimensional data
elif is_data_filtered == "low_dims":
    train_input = np.load("Data/low_dim_OK.npy")
    test_input = np.load("Data/low_dim_OK_extra.npy")
    is_data_filtered = "low_dims_"
# Otherwise load the full OK data
else:
    train_input = np.load("Data/OK.npy")
    test_input = np.load("Data/OK_extra.npy")

# Concat the test input to form full testing dataset
test_input = np.concatenate((train_input, test_input))

print(train_input.shape)
print(test_input.shape)

# Load the anomalies too
# Extended with plugged center part
if faulty_extended == "extended":
    if is_data_filtered == "BSE_":
        anomalous_input = np.load("Data/BSE_faulty_extended.npy")
    elif is_data_filtered == "SE_":
        anomalous_input = np.load("Data/SE_faulty_extended.npy")
    elif is_data_filtered == "low_dims_":
        anomalous_input = np.load("Data/low_dim_Faulty_extended.npy")
    else:
        anomalous_input = np.load("Data/Faulty_extended.npy")
    # Same thing is above with filtered data
    faulty_extended = "extended_"
# Or without the plugged center part
else:
    if is_data_filtered == "BSE_":
        anomalous_input = np.load("Data/BSE_faulty.npy")
    elif is_data_filtered == "SE_":
        anomalous_input = np.load("Data/SE_faulty.npy")
    elif is_data_filtered == "low_dims_":
        anomalous_input = np.load("Data/low_dim_Faulty.npy")
    else:
        anomalous_input = np.load("Data/Faulty.npy")

print(anomalous_input.shape)

# Arrays to hold reconstructed images for anomaly detection by mean squared error
reconstructed_ok_array = []
reconstructed_anomalous_array = []


# Choose desired model
model_class = globals()[desired_model]
model = model_class()


# Create and compile the model
input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)
model.create_net(input_shape)
model.compile_net()
# Train it
model.train_net(train_input, epochs=epochs, batch_size=batch_size)

# Plot the model's loss
plt.plot(model.history.history['loss'])
plt.title('model loss - ' + model.name)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('Graphs/Losses/' + str(is_data_filtered) + str(faulty_extended)
            + model.name + '_e' + str(epochs) + '_b' + str(batch_size)
            + '_loss.png', bbox_inches="tight")
plt.close('all')

# Save the weights and even the whole model
model.save_weights(epochs, batch_size, is_data_filtered, faulty_extended)
model.save_model(epochs, batch_size, is_data_filtered, faulty_extended)
# Save also the encoder part
model.save_encoder_model(epochs, batch_size, is_data_filtered, faulty_extended)

# To see the actual reconstructed images of the training and extra OK data
# And also to save them for MSE anomaly detection
for i in range(0, test_input.shape[0]):
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(test_input[i].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(IMG_WIDTH, IMG_HEIGHT)

    # Append the reconstructed image to the reconstructed_ok array
    reconstructed_ok_array.append(reconstructed_img)

    # Array has normalized values - need to multiply them again otherwise we get black picture
    im = Image.fromarray(reconstructed_img * 255.0)
    im = im.convert("L")
    im.save('Reconstructed/' + str(is_data_filtered) + model.name
            + '_e' + str(epochs) + '_b' + str(batch_size)
            + '_' + str(i) + '.jpg')

# Convert to numpy array
reconstructed_ok_array = np.array(reconstructed_ok_array)

# To see the actual reconstructed images of the anomalous data
# And also to save them for MSE anomaly detection
for i in range(0, anomalous_input.shape[0]):
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(anomalous_input[i].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(IMG_WIDTH, IMG_HEIGHT)

    # Append the reconstructed image to the reconstructed_ok array
    reconstructed_anomalous_array.append(reconstructed_img)

    # Array has normalized values - need to multiply them again otherwise we get black picture
    im = Image.fromarray(reconstructed_img * 255.0)
    im = im.convert("L")
    im.save('Reconstructed/' + str(is_data_filtered) + model.name
            + '_e' + str(epochs) + '_b' + str(batch_size)
            + '_' + str(i) + '_anomalous.jpg')

# Convert to numpy array
reconstructed_anomalous_array = np.array(reconstructed_anomalous_array)

# Array to hold MSE values
reconstructed_ok_errors = []
reconstructed_anomalous_errors = []

# Compute the reconstruction MSE for ok data
for i in range(0, test_input.shape[0]):
    # Reshape into 768x768
    original_image = test_input[i].reshape(IMG_WIDTH, IMG_HEIGHT)
    # Add reconstructed image MSE to the array
    reconstruction_error = np.square(np.subtract(original_image, reconstructed_ok_array[i])).mean()
    reconstructed_ok_errors.append(reconstruction_error)

# Same for the anomalous data
for i in range(0, anomalous_input.shape[0]):
    # Reshape into 768x768
    original_image = anomalous_input[i].reshape(IMG_WIDTH, IMG_HEIGHT)
    # Add reconstructed image MSE to the array
    reconstruction_error = np.square(np.subtract(original_image, reconstructed_anomalous_array[i])).mean()
    reconstructed_anomalous_errors.append(reconstruction_error)

#print(reconstructed_ok_errors)
#print(reconstructed_anomalous_errors)

# X axis coords representing indices of the individual images
x = range(0, len(reconstructed_ok_errors))
z = range(0 + len(reconstructed_ok_errors),
          len(reconstructed_anomalous_errors) + len(reconstructed_ok_errors))

# Plot the MSEs
plt.scatter(x, reconstructed_ok_errors, c='g', s=10,
            marker='o', edgecolors='black', label='OK')
plt.scatter(z, reconstructed_anomalous_errors, c='r', s=10,
            marker='o', edgecolors='black', label='Anomalous')
# Horizontal line at 3 times the standard deviation, typical for outlier detection
plt.axhline(y=(3 * np.std(reconstructed_ok_errors)), color='r', linestyle='-')
plt.legend(loc='upper left')
plt.title('model reconstruction error - ' + model.name)
plt.ylabel('Reconstruction Error')
plt.xlabel('Index')
plt.savefig('Graphs/ReconstructionErrors/' + str(is_data_filtered)
            + str(faulty_extended) + model.name + '_e' + str(epochs)
            + '_b' + str(batch_size) + '_RError.png', bbox_inches="tight")
plt.close('all')

# Save the error arrays too, so one can see which images were problematic
# Also useful for anomaly detction itself, as for each model the
# threshold for anomaly detection is based on these findings
# For example using 3 times the standard deviation of the
# OK errors array worked very well.
reconstructed_ok_errors = np.array(reconstructed_ok_errors)
reconstructed_anomalous_errors = np.array(reconstructed_anomalous_errors)

np.save('Reconstructed/Error_Arrays/' + str(is_data_filtered) +
        str(faulty_extended) +  model.name + '_e' + str(epochs) +
        '_b' + str(batch_size) + '_ROK.npy',
        reconstructed_ok_errors)

np.save('Reconstructed/Error_Arrays/' + str(is_data_filtered) +
        str(faulty_extended) +  model.name + '_e' + str(epochs) +
        '_b' + str(batch_size) + '_RAnomalous.npy',
        reconstructed_anomalous_errors)
