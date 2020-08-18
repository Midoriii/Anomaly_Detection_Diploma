import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from reshape_util import reshape_normalize
from Models.BasicAutoencoder import BasicAutoencoder
from Models.BasicAutoencoder_LF import BasicAutoencoder_LF
from Models.HighStrideAutoencoder import HighStrideAutoencoder
from Models.BasicAutoencoderDeeper import BasicAutoencoderDeeper
from Models.HighStrideAutoencoderDeeper import HighStrideAutoencoderDeeper
from Models.BasicAutoencoderEvenDeeper import BasicAutoencoderEvenDeeper
from Models.BasicAutoencoderEvenDeeperLLR import BasicAutoencoderEvenDeeperLLR
from Models.BasicAutoencoderEvenDeeperExtraLLR import BasicAutoencoderEvenDeeperExtraLLR
from Models.TransposeConvAutoencoder import TransposeConvAutoencoder
from Models.TransposeConvAutoencoderDeepExtraLLR import TransposeConvAutoencoderDeepExtraLLR
from Models.HighStrideTransposeConvAutoencoder import HighStrideTransposeConvAutoencoder
from Models.BasicAutoencoderLFDeeperLLR import BasicAutoencoderLFDeeperLLR
from Models.BasicAutoencoderHFDeeperLLR import BasicAutoencoderHFDeeperLLR
from Models.BasicAutoencoderEvenDeeperSuperLLR import BasicAutoencoderEvenDeeperSuperLLR




# Notes : Sources say to not use dropout .. also that batch_norm may be an overkill, and also
# to use simple ReLU ... dunno about padding='same' as well ..


img_width = 768
img_height = 768
input_shape = (img_width, img_height, 1)

# Default params
epochs = 2
batch_size = 1
desired_model = "BasicAutoencoder"
# Selection of full OK dataset or only filtered best ones
is_data_filtered = ""
# Selection of full Faulty dataset
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



# Load the previously stored data
# If filtered is desired, load those
if is_data_filtered == "filtered":
    data = np.load("Data/OK_filtered.npy")
    # Reshape to fit the desired input and Normalize the data
    train_input = reshape_normalize(data, img_width, img_height)
    # Add the underscore so that later graph and file
    # names contain the 'filtered_' prefix when expected
    is_data_filtered = "filtered_"
# If we want only the BSE Data
elif is_data_filtered == "BSE":
    data = np.load("Data/BSE_ok.npy")
    train_input = data
    is_data_filtered = "BSE_"
# If we want only the SE Data
elif is_data_filtered == "SE":
    data = np.load("Data/SE_ok.npy")
    train_input = data
    is_data_filtered = "SE_"
# Otherwise load the full OK data
else:
    part1 = np.load("Data/OK_1.npy")
    part2 = np.load("Data/OK_2.npy")
    data = np.concatenate((part1, part2))
    # Reshape to fit the desired input and Normalize the data
    train_input = reshape_normalize(data, img_width, img_height)

print(data.shape)


# Load the anomalies too
# Extended with plugged center part
if faulty_extended == "extended":
    if is_data_filtered == "BSE_":
        anomalies = np.load("Data/BSE_faulty_extended.npy")
        anomalous_input = anomalies
    elif is_data_filtered == "SE_":
        anomalies = np.load("Data/SE_faulty_extended.npy")
        anomalous_input = anomalies
    else:
        anomalies = np.load("Data/Faulty_extended.npy")
        # Reshape to fit the desired input and Normalize the data
        anomalous_input = reshape_normalize(anomalies, img_width, img_height)
    # Same thing is above with filtered data
    faulty_extended = "extended_"
# Or without the plugged center part
else:
    if is_data_filtered == "BSE_":
        anomalies = np.load("Data/BSE_faulty.npy")
        anomalous_input = anomalies
    elif is_data_filtered == "SE_":
        anomalies = np.load("Data/SE_faulty.npy")
        anomalous_input = anomalies
    else:
        anomalies = np.load("Data/Faulty.npy")
        # Reshape to fit the desired input and Normalize the data
        anomalous_input = reshape_normalize(anomalies, img_width, img_height)

print(anomalies.shape)

# Arrays to hold reconstructed images for anoamly detection by mean squared error
reconstructed_ok_array = []
reconstructed_anomalous_array = []


# Choose the correct model
if desired_model == "BasicAutoencoder":
    model = BasicAutoencoder()
elif desired_model == "BasicAutoencoder_LF":
    model = BasicAutoencoder_LF()
elif desired_model == "HighStrideAutoencoder":
    model = HighStrideAutoencoder()
elif desired_model == "HighStrideAutoencoderDeeper":
    model = HighStrideAutoencoderDeeper()
elif desired_model == "BasicAutoencoderDeeper":
    model = BasicAutoencoderDeeper()
elif desired_model == "BasicAutoencoderEvenDeeper":
    model = BasicAutoencoderEvenDeeper()
elif desired_model == "BasicAutoencoderEvenDeeperLLR":
    model = BasicAutoencoderEvenDeeperLLR()
elif desired_model == "BasicAutoencoderEvenDeeperExtraLLR":
    model = BasicAutoencoderEvenDeeperExtraLLR()
elif desired_model == "TransposeConvAutoencoder":
    model = TransposeConvAutoencoder()
elif desired_model == "HighStrideTransposeConvAutoencoder":
    model = HighStrideTransposeConvAutoencoder()
elif desired_model == "TransposeConvAutoencoderDeepExtraLLR":
    model = TransposeConvAutoencoderDeepExtraLLR()
elif desired_model == "BasicAutoencoderLFDeeperLLR":
    model = BasicAutoencoderLFDeeperLLR()
elif desired_model == "BasicAutoencoderHFDeeperLLR":
    model = BasicAutoencoderHFDeeperLLR()
elif desired_model == "BasicAutoencoderEvenDeeperSuperLLR":
    model = BasicAutoencoderEvenDeeperSuperLLR()
else:
    print("No model specified")
    sys.exit()

# Create and compile the model
model.create_net(input_shape)
model.compile_net()

# Train it
model.train_net(train_input, epochs=epochs, batch_size=batch_size)

plt.plot(model.history.history['loss'])
plt.title('model loss - ' + model.name)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('Graphs/Losses/' + str(is_data_filtered) + str(faulty_extended) + model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_loss.png', bbox_inches="tight")
plt.close('all')

# Save the weights and even the whole model
model.save_weights(epochs, batch_size, is_data_filtered, faulty_extended)
model.save_model(epochs, batch_size, is_data_filtered, faulty_extended)


# To see the actual reconstructed images of the training data
# And also to save them for MSE anomaly detection
for i in range(0, train_input.shape[0]):
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(train_input[i].reshape(1, img_width, img_height, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(img_width, img_height)

    # Append the reconstructed image to the reconstructed_ok array
    reconstructed_ok_array.append(reconstructed_img)

    # Array has normalized values - need to multiply them again otherwise we get black picture
    im = Image.fromarray(reconstructed_img * 255.0)
    im = im.convert("L")
    im.save('Reconstructed/' + str(is_data_filtered) + model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_' + str(i) + '.jpg')

# Convert to numpy array
reconstructed_ok_array = np.array(reconstructed_ok_array)

# To see the actual reconstructed images of the anomalous data
# And also to save them for MSE anomaly detection
for i in range(0, anomalous_input.shape[0]):
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(anomalous_input[i].reshape(1, img_width, img_height, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(img_width, img_height)

    # Append the reconstructed image to the reconstructed_ok array
    reconstructed_anomalous_array.append(reconstructed_img)

    # Array has normalized values - need to multiply them again otherwise we get black picture
    im = Image.fromarray(reconstructed_img * 255.0)
    im = im.convert("L")
    im.save('Reconstructed/' + str(is_data_filtered) + model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_' + str(i) + '_anomalous.jpg')

# Convert to numpy array
reconstructed_anomalous_array = np.array(reconstructed_anomalous_array)

# Array to hold MSE values
reconstructed_ok_errors = []
reconstructed_anomalous_errors = []

# Compute the reconstruction MSE for ok data
for i in range(0, train_input.shape[0]):
    # Reshape into 768x768
    original_image = train_input[i].reshape(img_width, img_height)
    # Add reconstructed image MSE to the array
    reconstructed_ok_errors.append(np.square(np.subtract(original_image, reconstructed_ok_array[i])).mean())

# Same for the anomalous data
for i in range(0, anomalous_input.shape[0]):
    # Reshape into 768x768
    original_image = anomalous_input[i].reshape(img_width, img_height)
    # Add reconstructed image MSE to the array
    reconstructed_anomalous_errors.append(np.square(np.subtract(original_image, reconstructed_anomalous_array[i])).mean())

#print(reconstructed_ok_errors)
#print(reconstructed_anomalous_errors)

# Plot the MSEs
x = range(0, len(reconstructed_ok_errors))
z = range(0 + len(reconstructed_ok_errors), len(reconstructed_anomalous_errors) + len(reconstructed_ok_errors))
plt.scatter(x, reconstructed_ok_errors, c='g', s=10, marker='o', edgecolors='black', label='OK')
plt.scatter(z, reconstructed_anomalous_errors, c='r', s=10, marker='o', edgecolors='black', label='Anomalous')
# Horizontal line at 3 times the standard deviation, typical for outlier detection
plt.axhline(y=(3 * np.std(reconstructed_ok_errors)), color='r', linestyle='-')
plt.legend(loc='upper left')
plt.title('model reconstruction error - ' + model.name)
plt.ylabel('Reconstruction Error')
plt.xlabel('Index')
plt.savefig('Graphs/ReconstructionErrors/' + str(is_data_filtered) + str(faulty_extended) + model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_RError.png', bbox_inches="tight")
plt.close('all')

# Save the error arrays too, so one can see which images were problematic
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
