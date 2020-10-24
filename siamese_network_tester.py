'''
The purpose of this script is to train various Siamese Nets on SE or BSE images
and test their anomaly detection performance.

Images of desired type (BSE vs SE, given by an argument -t) are loaded, and fed to
selected model (by an argument -m). Training happens for a number of epochs given
by an argument -e, and the loss used is selected by argument -l. Data of lower
dimensionality can also be selected by argument -d. Extended faulty data is
selected by argument -f.

After training, the model's loss and accuracy is plotted across the epochs and
saved for further reviewing. The model itself and its embedding part are also
saved. Embedding part corresponds to a single branch, which transforms input
into a feature vector.

Hand-picked prototypes are then loaded, and each OK and each Faulty image of
given type is tested against all 5 of the prototypes. Model.predict() returns a
value in range [0,1], where > 0.5 means the model predicts the images to be similar
and value <= 0.5 means the images are dissimilar. The sum of rounded predictions
for each image serves as prototype similarity score, where score of 5 means that the image
was predicted to be similar to all 5 prototypes and hence is considered OK,
score of 0 means an anomaly, since the image wasn't similar enough to the prototypes.
Anything in between is open for custom interpretation according to needs.

Graph overview of scores of individual images is also saved for quick performance
rating of the network. The final part of the script consist of some mock comparisons
which output confidence scores to standard otuput, they also serve preformance
reviewing purposes.


Arguments:
    -e / --epochs: The desired number of epochs the net should be trained for.
    -m / --model: Name of the model class to be instantiated and used.
    -t / --type: Accepts either 'BSE' or 'SE', the type of images to be used.
    -f / --faulty: If 'extended' given, all faulty images, plugged central hole
    included, are used.
    -l / --loss: Desired loss function for the model to use. Accepts: 'binary_crossentropy',
    'contrastive_loss'.
    -d / --dimensions: If 'low_dims' given, 384x384 images and prototypes are used
    instead. Low dimensional data is already using 'extended' faulty data, so
    this option requires the -f argument to be also set.
'''
from collections import Counter
from collections import OrderedDict

import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from Models.Siamese.BasicSiameseNet import BasicSiameseNet
from Models.Siamese.SiameseNetLF import SiameseNetLF
from Models.Siamese.SiameseNetDeeper import SiameseNetDeeper
from Models.Siamese.SiameseNetDeeperLLR import SiameseNetDeeperLLR
from Models.Siamese.SiameseNetMultipleConv import SiameseNetMultipleConv
from Models.Siamese.SiameseNetLite import SiameseNetLite
from Models.Siamese.SiameseNetLiteMultipleConv import SiameseNetLiteMultipleConv
from Models.Siamese.SiameseNetLiteMultipleConvAlt import SiameseNetLiteMultipleConvAlt
from Models.Siamese.SiameseNetLiteMultipleConvAltTwo import SiameseNetLiteMultipleConvAltTwo
from Models.Siamese.BasicSiameseNetLLR import BasicSiameseNetLLR
from Models.Siamese.BasicSiameseNetWoutBN import BasicSiameseNetWoutBN
from Models.Siamese.BasicSiameseNetWoutDropout import BasicSiameseNetWoutDropout
from Models.Siamese.BasicSiameseNetLowerDropout import BasicSiameseNetLowerDropout
from Models.Siamese.SiameseNetLiteMultipleConvLowerDropout import SiameseNetLiteMultipleConvLowerDropout
from Models.Siamese.SiameseNetLiteMultipleConvWithoutDropout import SiameseNetLiteMultipleConvWithoutDropout


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
        plt.text(rect.get_x() + rect.get_width()/2., 0.1 + height,
                 '%d' % int(height), color=color,
                 ha='center', va='bottom')

#Constants
IMG_WIDTH = 768
IMG_HEIGHT = 768

# Default params
epochs = 20
batch_size = 4
# Image type selection
image_type = "SE"
# Selection of faulty data - with or without plugged central holes
extended_faulty = ""
# The Model to be used
desired_model = "BasicSiameseNet"
# Loss for the siamese net to be used
desired_loss = "binary_crossentropy"
# String for when plots and weights are saved. Indicates the loss used,
# none loss mentioned = binary cross entropy, otherwise it's specified.
loss_string = ""
# String for when low dimensionality images are to be used
low_dims = ""

# Get full command-line arguments
full_cmd_arguments = sys.argv
# Keep all but the first
argument_list = full_cmd_arguments[1:]
# Getopt options
short_options = "e:m:t:f:l:d:"
long_options = ["epochs=", "model=", "type=", "faulty=", "loss=", "dimensions="]
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
    elif current_argument in ("-f", "--faulty"):
        extended_faulty = current_value
    elif current_argument in ("-l", "--loss"):
        desired_loss = current_value
        if current_value == "contrastive_loss":
            loss_string = "_ConLoss"
    elif current_argument in ("-d", "--dimensions"):
        low_dims = current_value

# Set the constants to 384x384 if low dimensional data is used
if low_dims == 'low_dims':
    IMG_WIDTH = 384
    IMG_HEIGHT = 384
    # Same old trick
    low_dims = "low_dims_"

# Loading data and labels - BSE or SE images as chosen in args
if image_type == 'BSE':
    if extended_faulty == 'extended':
        if low_dims == "low_dims_":
            left_data = np.load('DataHuge/low_dim_BSE_pairs_left_extended.npy')
            right_data = np.load('DataHuge/low_dim_BSE_pairs_right_extended.npy')
            labels = np.load('DataHuge/low_dim_BSE_pairs_labels_extended.npy')
        else:
            left_data = np.load('DataHuge/BSE_pairs_left_extended.npy')
            right_data = np.load('DataHuge/BSE_pairs_right_extended.npy')
            labels = np.load('DataHuge/BSE_pairs_labels_extended.npy')
        extended_faulty = "_extended"
    else:
        left_data = np.load('DataHuge/BSE_pairs_left.npy')
        right_data = np.load('DataHuge/BSE_pairs_right.npy')
        labels = np.load('DataHuge/BSE_pairs_labels.npy')

elif image_type == 'SE':
    if extended_faulty == 'extended':
        if low_dims == "low_dims_":
            left_data = np.load('DataHuge/low_dim_SE_pairs_left_extended.npy')
            right_data = np.load('DataHuge/low_dim_SE_pairs_right_extended.npy')
            labels = np.load('DataHuge/low_dim_SE_pairs_labels_extended.npy')
        else:
            left_data = np.load('DataHuge/SE_pairs_left_extended.npy')
            right_data = np.load('DataHuge/SE_pairs_right_extended.npy')
            labels = np.load('DataHuge/SE_pairs_labels_extended.npy')
        extended_faulty = "_extended"
    else:
        left_data = np.load('DataHuge/SE_pairs_left.npy')
        right_data = np.load('DataHuge/SE_pairs_right.npy')
        labels = np.load('DataHuge/SE_pairs_labels.npy')
else:
    print("Wrong Image Type specified!")
    sys.exit()
# Normalization not needed, data is already normalized


# Choose desired model
if desired_model == "BasicSiameseNet":
    model = BasicSiameseNet()
elif desired_model == "BasicSiameseNetLLR":
    model = BasicSiameseNetLLR()
elif desired_model == "BasicSiameseNetWoutBN":
    model = BasicSiameseNetWoutBN()
elif desired_model == "SiameseNetLF":
    model = SiameseNetLF()
elif desired_model == "SiameseNetDeeper":
    model = SiameseNetDeeper()
elif desired_model == "SiameseNetDeeperLLR":
    model = SiameseNetDeeperLLR()
elif desired_model == "SiameseNetMultipleConv":
    model = SiameseNetMultipleConv()
elif desired_model == "SiameseNetLite":
    model = SiameseNetLite()
elif desired_model == "SiameseNetLiteMultipleConv":
    model = SiameseNetLiteMultipleConv()
elif desired_model == "SiameseNetLiteMultipleConvAlt":
    model = SiameseNetLiteMultipleConvAlt()
elif desired_model == "SiameseNetLiteMultipleConvAltTwo":
    model = SiameseNetLiteMultipleConvAltTwo()
elif desired_model == "BasicSiameseNetWoutDropout":
    model = BasicSiameseNetWoutDropout()
elif desired_model == "BasicSiameseNetLowerDropout":
    model = BasicSiameseNetLowerDropout()
elif desired_model == "SiameseNetLiteMultipleConvLowerDropout":
    model = SiameseNetLiteMultipleConvLowerDropout()
elif desired_model == "SiameseNetLiteMultipleConvWithoutDropout":
    model = SiameseNetLiteMultipleConvWithoutDropout()
else:
    print("Wrong Model specified!")
    sys.exit()

# Create and compile the model
input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)
model.create_net(input_shape)
model.compile_net(desired_loss)
# Train it
model.train_net(left_data, right_data, labels, epochs=epochs, batch_size=batch_size)

# Plot loss and accuracy
plt.plot(model.history.history['loss'])
plt.title('model loss - ' + model.name)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('Graphs/Losses/' + str(low_dims) + model.name + '_' + str(image_type)
            + str(extended_faulty) + str(loss_string) + '_e' + str(epochs)
            + '_b' + str(batch_size) + '_loss.png', bbox_inches="tight")
plt.close('all')

plt.plot(model.history.history['binary_accuracy'])
plt.title('model accuracy - ' + model.name)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('Graphs/Accuracies/' + str(low_dims) + model.name + '_' + str(image_type)
            + str(extended_faulty) + str(loss_string) + '_e' + str(epochs)
            + '_b' + str(batch_size) + '_acc.png', bbox_inches="tight")
plt.close('all')

# Save the model and embedding model architecture
model.save_model(epochs, batch_size, image_type, extended_faulty, loss_string,
                 low_dims)
model.save_embedding_model(epochs, batch_size, image_type, extended_faulty, loss_string,
                           low_dims)

# For performance evaluation, load prototypes and actual data sorted by methods.
if image_type == 'SE':
    if low_dims == 'low_dims_':
        test_prototypes = np.load("Data/low_dim_SE_prototypes.npy")
        test_ok = np.load("Data/low_dim_SE_ok.npy")
        test_ok_extra = np.load("Data/low_dim_SE_ok_extra.npy")
    else:
        test_prototypes = np.load("Data/SE_prototypes.npy")
        test_ok = np.load("Data/SE_ok.npy")
        test_ok_extra = np.load("Data/SE_ok_extra.npy")

    if extended_faulty == '_extended':
        if low_dims == 'low_dims_':
            test_faulty = np.load("Data/low_dim_SE_faulty_extended.npy")
        else:
            test_faulty = np.load("Data/SE_faulty_extended.npy")
    else:
        test_faulty = np.load("Data/SE_faulty.npy")
else:
    if low_dims == 'low_dims_':
        test_prototypes = np.load("Data/low_dim_BSE_prototypes.npy")
        test_ok = np.load("Data/low_dim_BSE_ok.npy")
        test_ok_extra = np.load("Data/low_dim_BSE_ok_extra.npy")
    else:
        test_prototypes = np.load("Data/BSE_prototypes.npy")
        test_ok = np.load("Data/BSE_ok.npy")
        test_ok_extra = np.load("Data/BSE_ok_extra.npy")

    if extended_faulty == '_extended':
        if low_dims == 'low_dims_':
            test_faulty = np.load("Data/low_dim_BSE_faulty_extended.npy")
        else:
            test_faulty = np.load("Data/BSE_faulty_extended.npy")
    else:
        test_faulty = np.load("Data/BSE_faulty.npy")

# Concat the ok data to form the full training set
test_ok = np.concatenate((test_ok, test_ok_extra))
# Lists to save scores
prototype_similarity_scores_ok = []
prototype_similarity_scores_faulty = []

# For each okay image, get the score with each prototype
for sample in range(0, test_ok.shape[0]):
    score = 0
    for proto in range(0, test_prototypes.shape[0]):
        # It's rounded because we care only about 0s or 1s as predicted labels
        score += np.around(model.predict(test_ok[sample].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                                         test_prototypes[proto].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)))
    prototype_similarity_scores_ok.append(int(score))

# For each faulty image, get the score with each prototype
for sample in range(0, test_faulty.shape[0]):
    score = 0
    for proto in range(0, test_prototypes.shape[0]):
        score += np.around(model.predict(test_faulty[sample].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                                         test_prototypes[proto].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)))
    prototype_similarity_scores_faulty.append(int(score))


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
plt.savefig('Graphs/SiameseScores/' + str(low_dims) + model.name + "_" + str(image_type)
            + str(extended_faulty) + str(loss_string) + '_e' + str(epochs)
            + '_b' + str(batch_size) + '_AnoScore.png', bbox_inches="tight")
plt.close('all')

# Should be ~1
print("Expected: ~1")
print(model.predict(test_ok[2].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                    test_ok[5].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)))
# Should be ~0
print("Expected: ~0")
print(model.predict(test_faulty[2].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                    test_ok[5].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)))
# Should be ~0
print("Expected: ~0")
print(model.predict(test_ok[2].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                    test_faulty[5].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)))
# Should be ~0
print("Expected: ~0")
print(model.predict(test_ok[20].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                    test_faulty[15].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)))
# Should be ~1
print("Expected: ~1")
print(model.predict(test_ok[10].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                    test_ok[15].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)))
# Should be ~1
print("Expected: ~1")
print(model.predict(test_ok[12].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                    test_ok[100].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)))
# Should be ~1
print("Expected: ~1")
print(model.predict(test_faulty[2].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                    test_faulty[5].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)))
