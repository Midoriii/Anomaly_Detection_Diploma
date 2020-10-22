'''
bla
'''
from collections import Counter
from collections import OrderedDict

import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from Models.BasicTripletNet import BasicTripletNet
from Models.BasicTripletNetLite import BasicTripletNetLite
from Models.BasicTripletNetLiteWoutDropout import BasicTripletNetLiteWoutDropout
from Models.BasicTripletNetLowerDropout import BasicTripletNetLowerDropout
from Models.BasicTripletNetWoutDropout import BasicTripletNetWoutDropout
from Models.BasicTripletNetHLR import BasicTripletNetHLR
from Models.BasicTripletNetHLRLowerDropout import BasicTripletNetHLRLowerDropout
from Models.BasicTripletNetHLRWoutDropout import BasicTripletNetHLRWoutDropout
from Models.BasicTripletNetLLR import BasicTripletNetLLR
from Models.BasicTripletNetLLRWoutDropout import BasicTripletNetLLRWoutDropout
from Models.BasicTripletNetDeeper import BasicTripletNetDeeper
from Models.BasicTripletNetDeeperWoutDropout import BasicTripletNetDeeperWoutDropout
from Models.BasicTripletNetLF import BasicTripletNetLF
from Models.BasicTripletNetLFWoutDropout import BasicTripletNetLFWoutDropout
from Models.BasicTripletNetHF import BasicTripletNetHF
from Models.BasicTripletNetHFWoutDropout import BasicTripletNetHFWoutDropout
from Models.TripletNetMultipleConv import TripletNetMultipleConv
from Models.TripletNetMultipleConvWoutDropout import TripletNetMultipleConvWoutDropout
from Models.TripletNetLiteMultipleConv import TripletNetLiteMultipleConv
from Models.TripletNetLiteMultipleConvWoutDropout import TripletNetLiteMultipleConvWoutDropout


def autolabel(rects, color):
    """
    A helper function for barplot labeling. I chose to include this with the script
    to prevent additional file importing.

    Arguments:
        rects: A list of rectangles representing bar plots.
        color: Desired color of the labels.
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 0.25 + height,
                 '%d' % int(height), color=color,
                 ha='center', va='bottom')

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
elif desired_model == "BasicTripletNetLite":
    model = BasicTripletNetLite()
elif desired_model == "BasicTripletNetLiteWoutDropout":
    model = BasicTripletNetLiteWoutDropout()
elif desired_model == "BasicTripletNetLowerDropout":
    model = BasicTripletNetLowerDropout()
elif desired_model == "BasicTripletNetWoutDropout":
    model = BasicTripletNetWoutDropout()
elif desired_model == "BasicTripletNetHLR":
    model = BasicTripletNetHLR()
elif desired_model == "BasicTripletNetHLRLowerDropout":
    model = BasicTripletNetHLRLowerDropout()
elif desired_model == "BasicTripletNetHLRWoutDropout":
    model = BasicTripletNetHLRWoutDropout()
elif desired_model == "BasicTripletNetLLR":
    model = BasicTripletNetLLR()
elif desired_model == "BasicTripletNetLLRWoutDropout":
    model = BasicTripletNetLLRWoutDropout()
elif desired_model == "BasicTripletNetDeeper":
    model = BasicTripletNetDeeper()
elif desired_model == "BasicTripletNetDeeperWoutDropout":
    model = BasicTripletNetDeeperWoutDropout()
elif desired_model == "BasicTripletNetLF":
    model = BasicTripletNetLF()
elif desired_model == "BasicTripletNetLFWoutDropout":
    model = BasicTripletNetLFWoutDropout()
elif desired_model == "BasicTripletNetHF":
    model = BasicTripletNetHF()
elif desired_model == "BasicTripletNetHFWoutDropout":
    model = BasicTripletNetHFWoutDropout()
elif desired_model == "TripletNetMultipleConv":
    model = TripletNetMultipleConv()
elif desired_model == "TripletNetMultipleConvWoutDropout":
    model = TripletNetMultipleConvWoutDropout()
elif desired_model == "TripletNetLiteMultipleConv":
    model = TripletNetLiteMultipleConv()
elif desired_model == "TripletNetLiteMultipleConvWoutDropout":
    model = TripletNetLiteMultipleConvWoutDropout()
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


# Load data by image type for predictions
if image_type == "BSE":
    ok_data = np.load("Data/low_dim_BSE_ok.npy")
    ok_data_extra = np.load("Data/low_dim_BSE_ok_extra.npy")
    faulty_data = np.load("Data/low_dim_BSE_faulty_extended.npy")
    prototype_data = np.load("Data/low_dim_BSE_prototypes.npy")
else:
    ok_data = np.load("Data/low_dim_SE_ok.npy")
    ok_data_extra = np.load("Data/low_dim_SE_ok_extra.npy")
    faulty_data = np.load("Data/low_dim_SE_faulty_extended.npy")
    prototype_data = np.load("Data/low_dim_SE_prototypes.npy")

# Concat the ok data .. unlike in Siamese tester, where only the original OK data is used.
ok_data = np.concatenate((ok_data, ok_data_extra))
# Lists to save scores
prototype_similarity_scores_ok = []
prototype_similarity_scores_faulty = []

# For each okay image, get the score with each prototype
for sample in range(0, ok_data.shape[0]):
    score = 0
    sample_prediction = model.predict(ok_data[sample].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
    for proto in range(0, prototype_data.shape[0]):
        prototype_prediction = model.predict(prototype_data[proto].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
        # Calculate the distance
        distance = np.sum(np.square(sample_prediction - prototype_prediction))
        # If distance is over 5.0 (for margin 10.0), it's most likely an anomaly.
        if distance < 5.0:
            score += 1
    prototype_similarity_scores_ok.append(score)

# For each faulty image, get the score with each prototype
for sample in range(0, faulty_data.shape[0]):
    score = 0
    sample_prediction = model.predict(faulty_data[sample].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
    for proto in range(0, prototype_data.shape[0]):
        prototype_prediction = model.predict(prototype_data[proto].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
        # Calculate the distance
        distance = np.sum(np.square(sample_prediction - prototype_prediction))
        # If distance is over 5.0 (for margin 10.0), it's most likely an anomaly.
        if distance < 5.0:
            score += 1
    prototype_similarity_scores_faulty.append(score)


# Use Counters to get the total # of images by each score
ok_counter = Counter(prototype_similarity_scores_ok)
anomalous_counter = Counter(prototype_similarity_scores_faulty)
# Make ordered dictionaries out of the Counters, for plotting purposes, sorted
# in ascending order; from 0 to 5
ok_ordered = OrderedDict(sorted(ok_counter.items(), key=lambda x: x[0]))
anomalous_ordered = OrderedDict(sorted(anomalous_counter.items(), key=lambda x: x[0]))

# Set colors for graphs based on image type
if image_type == 'BSE':
    graph_color = 'tab:blue'
else:
    graph_color = 'tab:green'

# X axis coords representing prototype similarity scores; 0-5
X = np.arange(6)

# Plot the results
ok_bars = plt.bar(X - 0.10, ok_ordered.values(), color=graph_color, width=0.20,
                  label="OK")
anomalous_bars = plt.bar(X + 0.10, anomalous_ordered.values(), color='tab:red', width=0.20,
                         label="Anomalous")
plt.legend(loc='upper center')
plt.title('Model ' + model.name + "_" + image_type)
plt.xlabel('Prototype similarity score')
plt.ylabel('Image count')
autolabel(ok_bars, graph_color)
autolabel(anomalous_bars, 'tab:red')
plt.savefig('Graphs/TripletScores/' + dimensions + model.name + "_" + str(image_type)
            + "_set_" + image_set + '_e' + str(epochs) + '_b' + str(batch_size)
            + '_AnoScore.png', bbox_inches="tight")
plt.close('all')


# Try some predictions
test_anchor = model.predict(ok_data[24].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
test_prototype = model.predict(ok_data[48].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
print("OK and OK:")
print(np.sum(np.square(test_anchor - test_prototype)))

test_anchor = model.predict(ok_data[2].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
test_prototype = model.predict(ok_data[86].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
print("OK and OK:")
print(np.sum(np.square(test_anchor - test_prototype)))

test_anchor = model.predict(ok_data[37].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
test_prototype = model.predict(faulty_data[7].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
print("OK and Faulty:")
print(np.sum(np.square(test_anchor - test_prototype)))

test_anchor = model.predict(faulty_data[14].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
test_prototype = model.predict(faulty_data[5].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
print("Faulty and Faulty:")
print(np.sum(np.square(test_anchor - test_prototype)))

test_anchor = model.predict(faulty_data[11].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
test_prototype = model.predict(ok_data[5].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
print("Faulty and OK:")
print(np.sum(np.square(test_anchor - test_prototype)))

test_anchor = model.predict(faulty_data[17].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
test_prototype = model.predict(ok_data[55].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
print("Faulty and OK:")
print(np.sum(np.square(test_anchor - test_prototype)))

test_anchor = model.predict(faulty_data[11].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
test_prototype = model.predict(faulty_data[8].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
print("Faulty and Faulty:")
print(np.sum(np.square(test_anchor - test_prototype)))
