'''
bla
'''
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf

from Models.BasicTripletNet import BasicTripletNet
from Models.BasicTripletNetHLR import BasicTripletNetHLR
from Models.BasicTripletNetLLR import BasicTripletNetLLR
from Models.BasicTripletNetDeeper import BasicTripletNetDeeper
from Models.TripletNetMultipleConv import TripletNetMultipleConv


# To allows tensor unpacking
#tf.enable_eager_execution()
# Constants
IMG_WIDTH = 384
IMG_HEIGHT = 384

# Default params
epochs = 5
batch_size = 4
dimensions = "low_dim_"
# Image type selection
image_type = "SE"
# Data set selection (1,2,3)
image_set = "1"
# The Model to be used
desired_model = "BasicTripletNet"

# Get full command-line arguments
full_cmd_arguments = sys.argv
# Keep all but the first
argument_list = full_cmd_arguments[1:]
# Getopt options
short_options = "e:m:t:s:"
long_options = ["epochs=", "model=", "type=", "imset="]
# Get the arguments and their respective values
arguments, values = getopt.getopt(argument_list, short_options, long_options)

# Evaluate given options
for current_argument, current_value in arguments:
    if current_argument in ("-e", "--epochs"):
        epochs = int(current_value)
    elif current_argument in ("-m", "--model"):
        desired_model = current_value
    elif current_argument in ("-t", "--type"):
        image_type = current_value
    elif current_argument in ("-s", "--imset"):
        image_set = str(current_value)


# Load appropriate data, selected by image type and image set number
if image_type == "BSE":
    anchor_data = np.load("DataTriplet/low_dim_BSE_triplet_anchor_" + image_set + ".npy")
    pos_data = np.load("DataTriplet/low_dim_BSE_triplet_pos_" + image_set + ".npy")
    neg_data = np.load("DataTriplet/low_dim_BSE_triplet_neg_" + image_set + ".npy")
elif image_type == "SE":
    anchor_data = np.load("DataTriplet/low_dim_SE_triplet_anchor_" + image_set + ".npy")
    pos_data = np.load("DataTriplet/low_dim_SE_triplet_pos_" + image_set + ".npy")
    neg_data = np.load("DataTriplet/low_dim_SE_triplet_neg_" + image_set + ".npy")
else:
    print("Wrong Image Type specified!")
    sys.exit()


# Choose desired model
if desired_model == "BasicTripletNet":
    model = BasicTripletNet()
elif desired_model == "BasicTripletNetHLR":
    model = BasicTripletNetHLR()
elif desired_model == "BasicTripletNetLLR":
    model = BasicTripletNetLLR()
elif desired_model == "BasicTripletNetDeeper":
    model = BasicTripletNetDeeper()
elif desired_model == "TripletNetMultipleConv":
    model = TripletNetMultipleConv()
else:
    print("Wrong Model specified!")
    sys.exit()


# Create and compile the model
input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)
model.create_net(input_shape)
model.compile_net()
# Train it
model.train_net(anchor_data, pos_data, neg_data, epochs=epochs, batch_size=batch_size)

# Plot loss
plt.plot(model.history.history['loss'])
plt.title('model loss - ' + model.name)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('Graphs/Losses/' + dimensions + model.name + '_' + str(image_type)
            + "_set_" + image_set + '_e' + str(epochs) + '_b' + str(batch_size)
            + '_loss.png', bbox_inches="tight")
plt.close('all')

# Save the model and embedding model architecture
model.save_model(epochs, batch_size, image_type, image_set, dimensions)
model.save_embedding_model(epochs, batch_size, image_type, image_set, dimensions)
