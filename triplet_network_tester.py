'''
Copyright (c) 2021, Štěpán Beneš


The purpose of this script is to train various TripletScores Nets on SE or BSE images
and test their anomaly detection performance.

Images of desired type (BSE vs SE, given by an argument -t) are loaded, and fed to
selected model (by an argument -m). Training happens for a number of epochs given
by an argument -e. Data is already low dimensional (384x384) only. As triplet data
was generated randomly (3 sets), the desired set can be chosen by argument -s.
Multiple sets were made to test the role of randomness, as unlike in Siamese nets,
images can't sipmly be paired each with each, but the triplets have to be randomly
sampled.

After training, the model's loss is plotted across the epochs and saved for further
reviewing. The model itself and its embedding part are also saved. Embedding part
corresponds to a single branch, which transforms input into a feature vector.

Hand-picked prototypes are then loaded, and each OK and each Faulty image of
given type is tested against all 5 of the prototypes. Model.predict() returns an
embedding, and the embeddings of tested images and prototypes are then evaluated
using MSE.

The networks are trained to learn a margin of 10 between OK and Defective images,
but in practice a threshold of 5 is perfectly fine to decide if image is anomalous
or not, even though the best models really learn to classify OK as 0.0 and Faulty
as 10.0.

So the MSE is evaluated like this: distance of over 5.0 means that the images
are dissimilar - using only OK prototypes, this means that the tested image is
defective. Distance under 5.0 means that the images are similar enough and the
tested image is OK. As with Siamese nets, each image is given prototype similarity
score, where score of 5 means that the image was predicted to be similar to all
5 prototypes and hence is considered OK, score of 0 means an anomaly, since
the image wasn't similar enough to the prototypes. Anything in between is open
for custom interpretation according to needs.

Graph overview of how many OK/Defective images ended up with each score
is also saved for quick performance rating of the network. The final part of the script
consist of some mock comparisons which output confidence scores to standard otuput,
they also serve preformance reviewing purposes.


Arguments:
    -e / --epochs: The desired number of epochs the net should be trained for.
    -m / --model: Name of the model class to be instantiated and used.
    -t / --type: Accepts either 'BSE' or 'SE', the type of images to be used.
    -s / --imset: The desired dataset number - 1, 2, 3.
'''
from collections import Counter
from collections import OrderedDict

import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from Models.Triplet.BasicTripletNet import BasicTripletNet
from Models.Triplet.BasicTripletNetLite import BasicTripletNetLite
from Models.Triplet.BasicTripletNetLiteWoutDropout import BasicTripletNetLiteWoutDropout
from Models.Triplet.BasicTripletNetLowerDropout import BasicTripletNetLowerDropout
from Models.Triplet.BasicTripletNetWoutDropout import BasicTripletNetWoutDropout
from Models.Triplet.BasicTripletNetHLR import BasicTripletNetHLR
from Models.Triplet.BasicTripletNetHLRLowerDropout import BasicTripletNetHLRLowerDropout
from Models.Triplet.BasicTripletNetHLRWoutDropout import BasicTripletNetHLRWoutDropout
from Models.Triplet.BasicTripletNetLLR import BasicTripletNetLLR
from Models.Triplet.BasicTripletNetLLRWoutDropout import BasicTripletNetLLRWoutDropout
from Models.Triplet.BasicTripletNetDeeper import BasicTripletNetDeeper
from Models.Triplet.BasicTripletNetDeeperWoutDropout import BasicTripletNetDeeperWoutDropout
from Models.Triplet.BasicTripletNetLF import BasicTripletNetLF
from Models.Triplet.BasicTripletNetLFWoutDropout import BasicTripletNetLFWoutDropout
from Models.Triplet.BasicTripletNetHF import BasicTripletNetHF
from Models.Triplet.BasicTripletNetHFWoutDropout import BasicTripletNetHFWoutDropout
from Models.Triplet.TripletNetMultipleConv import TripletNetMultipleConv
from Models.Triplet.TripletNetMultipleConvWoutDropout import TripletNetMultipleConvWoutDropout
from Models.Triplet.TripletNetLiteMultipleConv import TripletNetLiteMultipleConv
from Models.Triplet.TripletNetLiteMultipleConvWoutDropout import TripletNetLiteMultipleConvWoutDropout


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
        if height > 0:
            plt.text(rect.get_x() + rect.get_width()/2., 0.1 + height,
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
model_class = globals()[desired_model]
model = model_class()


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
# Fill any possible missing key values between 0-5 with zeroes, for the plots to work
for i in range(0, 6):
    if not ok_counter.get(i):
        ok_counter[i] = 0
    if not anomalous_counter.get(i):
        anomalous_counter[i] = 0
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
ok_bars = plt.bar(X + 0.20, ok_ordered.values(), color=graph_color, width=0.40,
                  label="Without Defect")
anomalous_bars = plt.bar(X - 0.20, anomalous_ordered.values(), color='tab:red', width=0.40,
                         label="Defective")
plt.legend(loc='upper center')
plt.title('Model ' + model.name + " " + image_type)
plt.xlabel('Prototype similarity score')
plt.ylabel('Image count')
plt.ylim(0, 200)
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
