'''
The purpose of this script is to train various Siamese Nets on SE or BSE images
and test their anomaly detection performance.

Images of desired type (BSE vs SE, given by an argument -t) are loaded, and fed to
selected model (by an argument -m). Training happens for a number of epochs given
by an argument -e, and the loss used is selected by argument -l.

After training, the model's loss and accuracy is plotted across the epochs and
saved for further reviewing. The model itself and its embedding part are also
saved. Embedding part corresponds to a single branch, which transforms input
into a feature vector.

Hand-picked prototypes are then loaded, and each OK and each Faulty image of
given type is tested against all 5 of the prototypes. Model.predict() returns a
value in range [0,1], where > 0.5 means the model predicts the images to be similar
and value <= 0.5 means the images are dissimilar. The sum of rounded predictions
for each image serves as anomaly score, where score of 5 means that the image
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
    'triplet_loss', 'contrastive_loss'.
'''
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from Models.BasicSiameseNet import BasicSiameseNet
from Models.SiameseNetLF import SiameseNetLF
from Models.SiameseNetDeeper import SiameseNetDeeper
from Models.SiameseNetDeeperLLR import SiameseNetDeeperLLR
from Models.SiameseNetMultipleConv import SiameseNetMultipleConv
from Models.SiameseNetLite import SiameseNetLite
from Models.SiameseNetLiteMultipleConv import SiameseNetLiteMultipleConv
from Models.SiameseNetLiteMultipleConvAlt import SiameseNetLiteMultipleConvAlt
from Models.SiameseNetLiteMultipleConvAltTwo import SiameseNetLiteMultipleConvAltTwo
from Models.BasicSiameseNetLLR import BasicSiameseNetLLR
from Models.BasicSiameseNetWoutBN import BasicSiameseNetWoutBN
from Models.BasicSiameseNetWoutDropout import BasicSiameseNetWoutDropout
from Models.BasicSiameseNetLowerDropout import BasicSiameseNetLowerDropout
from Models.SiameseNetLiteMultipleConvLowerDropout import SiameseNetLiteMultipleConvLowerDropout
from Models.SiameseNetLiteMultipleConvWithoutDropout import SiameseNetLiteMultipleConvWithoutDropout


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

# Get full command-line arguments
full_cmd_arguments = sys.argv
# Keep all but the first
argument_list = full_cmd_arguments[1:]
# Getopt options
short_options = "e:m:t:f:l:"
long_options = ["epochs=", "model=", "type=", "faulty=", "loss="]
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


# Loading data and labels - BSE or SE images as chosen in args
if image_type == 'BSE':
    if extended_faulty == 'extended':
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
plt.savefig('Graphs/Losses/' + model.name + '_' + str(image_type)
            + str(extended_faulty) + str(loss_string) + '_e' + str(epochs)
            + '_b' + str(batch_size) + '_loss.png', bbox_inches="tight")
plt.close('all')

plt.plot(model.history.history['binary_accuracy'])
plt.title('model accuracy - ' + model.name)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('Graphs/Accuracies/' + model.name + '_' + str(image_type)
            + str(extended_faulty) + str(loss_string) + '_e' + str(epochs)
            + '_b' + str(batch_size) + '_acc.png', bbox_inches="tight")
plt.close('all')

# Save the model and embedding model architecture
model.save_model(epochs, batch_size, image_type, extended_faulty, loss_string)
model.save_embedding_model(epochs, batch_size, image_type, extended_faulty, loss_string)

# For performance evaluation, load prototypes and each image and get anomaly score
# Load prototypes and actual data sorted by methods
if image_type == 'SE':
    test_prototypes = np.load("Data/SE_prototypes.npy")
    test_ok = np.load("Data/SE_ok.npy")
    if extended_faulty == '_extended':
        test_faulty = np.load("Data/SE_faulty_extended.npy")
    else:
        test_faulty = np.load("Data/SE_faulty.npy")
else:
    test_prototypes = np.load("Data/BSE_prototypes.npy")
    test_ok = np.load("Data/BSE_ok.npy")
    if extended_faulty == '_extended':
        test_faulty = np.load("Data/BSE_faulty_extended.npy")
    else:
        test_faulty = np.load("Data/BSE_faulty.npy")
# Lists to save scores
anomaly_scores_ok = []
anomaly_scores_faulty = []

# For each okay image, get the score with each prototype
for sample in range(0, test_ok.shape[0]):
    score = 0
    for proto in range(0, test_prototypes.shape[0]):
        # It's rounded because we care only about 0s or 1s as predicted labels
        score += np.around(model.predict(test_ok[sample].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                                         test_prototypes[proto].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)))
    anomaly_scores_ok.append(score)

# For each faulty image, get the score with each prototype
for sample in range(0, test_faulty.shape[0]):
    score = 0
    for proto in range(0, test_prototypes.shape[0]):
        score += np.around(model.predict(test_faulty[sample].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                                         test_prototypes[proto].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)))
    anomaly_scores_faulty.append(score)

anomaly_scores_ok = np.array(anomaly_scores_ok)
anomaly_scores_faulty = np.array(anomaly_scores_faulty)

# Set colors for graphs based on image type
if image_type == 'BSE':
    scatter_color = 'b'
else:
    scatter_color = 'g'

# X axis coords representing indices of the individual images
x = range(0, len(anomaly_scores_ok))
z = range(0 + len(anomaly_scores_ok),
          len(anomaly_scores_ok) + len(anomaly_scores_faulty))

# Plot the resulting numbers stored in array
plt.scatter(x, anomaly_scores_ok, c=scatter_color,
            s=10, marker='o', edgecolors='black', label='OK')
plt.scatter(z, anomaly_scores_faulty, c='r',
            s=10, marker='o', edgecolors='black', label='Anomalous')
plt.legend(loc='upper left')
plt.title('Model ' + model.name + "_" + image_type)
plt.yticks(np.arange(0.0, 6.0, 1.0))
plt.ylabel('Anomaly Score')
plt.xlabel('Index')
plt.savefig('Graphs/SiameseScores/' + model.name + "_" + str(image_type)
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
