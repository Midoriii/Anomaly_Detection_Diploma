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
from Models.BasicSiameseNetLLR import BasicSiameseNetLLR
from Models.BasicSiameseNetWoutBN import BasicSiameseNetWoutBN
from Models.BasicSiameseNetWoutDropout import BasicSiameseNetWoutDropout
from Models.BasicSiameseNetLowerDropout import BasicSiameseNetLowerDropout
from Models.SiameseNetLiteMultipleConvLowerDropout import SiameseNetLiteMultipleConvLowerDropout
from Models.SiameseNetLiteMultipleConvWithoutDropout import SiameseNetLiteMultipleConvWithoutDropout


img_width = 768
img_height = 768
input_shape = (img_width, img_height, 1)
# Default params
epochs = 20
batch_size = 4
image_type = 'SE'
extended_faulty = ""
desired_model = "BasicSiameseNet"

# Get full command-line arguments
full_cmd_arguments = sys.argv
# Keep all but the first
argument_list = full_cmd_arguments[1:]
# Getopt options
short_options = "e:m:t:f:"
long_options = ["epochs=", "model=", "type=", "faulty="]
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


# Loading data and labels - BSE or SE image origin as chosen in args
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
model.create_net(input_shape)
model.compile_net()
# Train it
model.train_net(left_data, right_data, labels, epochs=epochs, batch_size=batch_size)

# Plot loss and accuracy
plt.plot(model.history.history['loss'])
plt.title('model loss - ' + model.name)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('Graphs/Losses/' + model.name + '_' + str(image_type) + str(extended_faulty) + '_e' + str(epochs) + '_b' + str(batch_size) + '_loss.png', bbox_inches="tight")
plt.close('all')

plt.plot(model.history.history['binary_accuracy'])
plt.title('model accuracy - ' + model.name)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('Graphs/Accuracies/' + model.name + '_' + str(image_type) + str(extended_faulty) + '_e' + str(epochs) + '_b' + str(batch_size) + '_acc.png', bbox_inches="tight")
plt.close('all')

# Save the model weights and architecture
model.save_model(epochs, batch_size, image_type, extended_faulty)
model.save_embedding_model(epochs, batch_size, image_type, extended_faulty)

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
# List to save scores
anomaly_scores_ok = []
anomaly_scores_faulty = []

# For each okay image, get the score with each prototype
for sample in range(0, test_ok.shape[0]):
    score = 0
    for proto in range(0, test_prototypes.shape[0]):
        score += np.around(model.predict(test_ok[sample].reshape(1, img_width, img_height, 1),
                                         test_prototypes[proto].reshape(1, img_width, img_height, 1)))
    anomaly_scores_ok.append(score)

# For each faulty image, get the score with each prototype
for sample in range(0, test_faulty.shape[0]):
    score = 0
    for proto in range(0, test_prototypes.shape[0]):
        score += np.around(model.predict(test_faulty[sample].reshape(1, img_width, img_height, 1),
                                         test_prototypes[proto].reshape(1, img_width, img_height, 1)))
    anomaly_scores_faulty.append(score)

anomaly_scores_ok = np.array(anomaly_scores_ok)
anomaly_scores_faulty = np.array(anomaly_scores_faulty)

# Set colors for graphs based on image type
if image_type == 'BSE':
    scatter_color = 'b'
else:
    scatter_color = 'g'

x = range(0, len(anomaly_scores_ok))
z = range(0 + len(anomaly_scores_ok), len(anomaly_scores_ok) + len(anomaly_scores_faulty))

# Plot the resulting numbers stored in array
plt.scatter(x, anomaly_scores_ok, c=scatter_color, s=10, marker='o', edgecolors='black', label='OK')
plt.scatter(z, anomaly_scores_faulty, c='r', s=10, marker='o', edgecolors='black', label='Anomalous')
plt.legend(loc='upper left')
plt.title('Model ' + model.name + "_" + image_type)
plt.yticks(np.arange(0.0, 6.0, 1.0))
plt.ylabel('Anomaly Score')
plt.xlabel('Index')
plt.savefig('Graphs/SiameseScores/' + model.name + "_" + str(image_type) + str(extended_faulty) + '_e' + str(epochs) + '_b' + str(batch_size) + '_AnoScore.png', bbox_inches="tight")
plt.close('all')

print(model.predict(test_ok[2].reshape(1, img_width, img_height, 1), test_ok[5].reshape(1, img_width, img_height, 1)))
print(model.predict(test_faulty[2].reshape(1, img_width, img_height, 1), test_ok[5].reshape(1, img_width, img_height, 1)))
print(model.predict(test_ok[2].reshape(1, img_width, img_height, 1), test_faulty[5].reshape(1, img_width, img_height, 1)))
print(model.predict(test_ok[20].reshape(1, img_width, img_height, 1), test_faulty[15].reshape(1, img_width, img_height, 1)))
print(model.predict(test_ok[10].reshape(1, img_width, img_height, 1), test_ok[15].reshape(1, img_width, img_height, 1)))
print(model.predict(test_ok[12].reshape(1, img_width, img_height, 1), test_ok[100].reshape(1, img_width, img_height, 1)))
print(model.predict(test_faulty[2].reshape(1, img_width, img_height, 1), test_faulty[5].reshape(1, img_width, img_height, 1)))
