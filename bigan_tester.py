'''
bla
'''
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from Models.biGAN.BasicBigan import BasicBigan


# Constants
IMG_WIDTH = 384
IMG_HEIGHT = 384

# Default params
epochs = 5
batch_size = 32
dimensions = "low_dim_"
# Image type selection
image_type = "SE"
# The Model to be used
desired_model = "BasicBigan"

# Get full command-line arguments
full_cmd_arguments = sys.argv
# Keep all but the first
argument_list = full_cmd_arguments[1:]
# Getopt options
short_options = "e:b:m:t:"
long_options = ["epochs=", "batch_size=", "model=", "type="]
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
    elif current_argument in ("-t", "--type"):
        image_type = current_value


# Load appropriate data, selected by image type
if image_type == "BSE":
    train_input = np.load("Data/low_dim_BSE_ok.npy")
    test_input = np.load("Data/low_dim_BSE_ok_extra.npy")
    test_input = np.concatenate((train_input, test_input))
    anomalous_input = np.load("Data/low_dim_BSE_faulty_extended.npy")
elif image_type == "SE":
    train_input = np.load("Data/low_dim_SE_ok.npy")
    test_input = np.load("Data/low_dim_SE_ok_extra.npy")
    test_input = np.concatenate((train_input, test_input))
    anomalous_input = np.load("Data/low_dim_SE_faulty_extended.npy")
else:
    print("Wrong Image Type specified!")
    sys.exit()


# Choose desired model
if desired_model == "BasicBigan":
    model = BasicBigan(IMG_WIDTH, batch_size=batch_size)
else:
    print("Wrong Model specified!")
    sys.exit()


# Train it
model.train(train_input, epochs=epochs)


# Try Predictions
verdict = model.predict(train_input[12])
print("OK:")
print("Score: " + str(verdict))

verdict = model.predict(test_input[29])
print("OK:")
print("Score: " + str(verdict))

verdict = model.predict(train_input[80])
print("OK:")
print("Score: " + str(verdict))

verdict = model.predict(anomalous_input[12])
print("Defective:")
print("Score: " + str(verdict))

verdict = model.predict(anomalous_input[5])
print("Defective:")
print("Score: " + str(verdict))
