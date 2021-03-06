'''
Copyright (c) 2021, Štěpán Beneš


The purpose of this script is to test various Variational Autoencoder models on provided
images.

The type of images to use for training and testing is given by the parameters -t
and the model to be trained and tested is given by -m. The number of training epochs
by -e and the batch size to be used by -b. The images used are exclusively low
dimensional (384x384).

The model is trained only on the original OK images. After training is done, the
model's loss is plotted to a graph and saved. The model itself and the weights
are saved too. Unlike simple Autoencoders, there's no real Encoder part to be saved.

For performance evaluation, the original OK, extra OK, and Faulty images are served
to the trained model (using Model.predict()) and the reconstructions are saved
to OK and Faulty lists. Afterwards, for each image, the MSE of the difference
of the original image and its reconstruction is computed and serves as a 'score' of sorts.

The idea is that a large reconstruction error points to an anomaly, as the model
is trained only on the OK data, and should struggle to reconstruct anomalous input.

To visualize the performance a graph is plotted of images and their respective
reconstruction errors. Green dots = truly OK images, red dots = truly Faulty images.

Common practice for finding a threshold for anomaly detection is to use 3 times the
standard deviation of OK scores. Such threshold is also shown on the graph and
serves as a divider of sorts showing which images are possibly faulty and which are OK.
Of course any such threshold can be edited to better serve FP vs FN needs.

Several reconstructed images are also saved, to gain a better insight into the
performance of the models. Some mock predictions also happen at the end of the
scripts, with the reconstruction errors printed out to standard output. These
serve as a quick performance check.


Arguments:
    -e / --epochs: Desired number of epochs for the model to train for.
    -b / --batch_size: Training batch_size to be used, can't handle more than 16
    reliably on most GPUs due to VRAM limits.
    -m / --model: Name of the model class to be instantiated and used.
    -t / --type: Type of data to be used, BSE or SE.
'''
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from Models.VAE.BasicVAE import BasicVAE
from Models.VAE.BasicVAE_HF import BasicVAE_HF
from Models.VAE.BasicVAE_IF import BasicVAE_IF
from Models.VAE.BasicVAEDeeper import BasicVAEDeeper
from Models.VAE.BasicVAEDeeper_IF import BasicVAEDeeper_IF
from Models.VAE.BasicVAEDeeper_LLR import BasicVAEDeeper_LLR
from Models.VAE.BasicVAEDeeper_LowLatDim import BasicVAEDeeper_LowLatDim
from Models.VAE.BasicVAE_HLR import BasicVAE_HLR
from Models.VAE.BasicVAE_LLR import BasicVAE_LLR
from Models.VAE.BasicVAE_HiLatDim import BasicVAE_HiLatDim
from Models.VAE.BasicVAE_ExtraHiLatDim import BasicVAE_ExtraHiLatDim
from Models.VAE.BasicVAE_LowLatDim import BasicVAE_LowLatDim
from Models.VAE.BasicVAE_HiLatDim_LLR import BasicVAE_HiLatDim_LLR
from Models.VAE.BasicVAE_LowLatDim_LLR import BasicVAE_LowLatDim_LLR
from Models.VAE.BasicVAE_LowRLFactor import BasicVAE_LowRLFactor
from Models.VAE.BasicVAE_HiRLFactor import BasicVAE_HiRLFactor
from Models.VAE.BasicVAE_HiRLFactor_LowLatDim import BasicVAE_HiRLFactor_LowLatDim
from Models.VAE.BasicVAE_ExtraHiRLFactor import BasicVAE_ExtraHiRLFactor
from Models.VAE.TransposeVAE import TransposeVAE
from Models.VAE.TransposeVAEDeeper import TransposeVAEDeeper
from Models.VAE.TransposeVAE_HiRLFactor_LowLatDim import TransposeVAE_HiRLFactor_LowLatDim


# Constants
IMG_WIDTH = 384
IMG_HEIGHT = 384

# Default params
epochs = 5
batch_size = 16
dimensions = "low_dim_"
# Image type selection
image_type = "SE"
# The Model to be used
desired_model = "BasicVAE"

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
model_class = globals()[desired_model]
model = model_class(IMG_WIDTH)


# Train it
model.train_net(train_input, epochs=epochs, batch_size=batch_size)
# Save it
model.save_model(epochs, image_type, dimensions)
# Save the weights for predictor
model.save_weights(epochs, image_type, dimensions)


# Plot the model's loss
plt.plot(model.history.history['loss'])
plt.title('model loss - ' + model.name)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('Graphs/Losses/vae_' + str(dimensions) + model.name + "_" + str(image_type)
            + '_e' + str(epochs) + '_b' + str(batch_size) + '_loss.png', bbox_inches="tight")
plt.close('all')


# Calculate Anomaly scores for test input
test_scores = []
for i in range(0, test_input.shape[0]):
    test_scores.append(model.predict(test_input[i]))

# Calculate Anomaly scores for anomalous input
anomalous_scores = []
for i in range(0, anomalous_input.shape[0]):
    anomalous_scores.append(model.predict(anomalous_input[i]))

# Define simple Anomaly detection threshold
threshold = 3 * np.std(test_scores[:train_input.shape[0]])

# Plot and save the anomaly scores
X1 = np.arange(0, len(test_scores))
X2 = np.arange(len(test_scores), len(test_scores) + len(anomalous_scores))

plt.scatter(X1, test_scores, c='g', s=10,
            marker='o', edgecolors='black', label='Without Defect')
plt.scatter(X2, anomalous_scores, c='r', s=10,
            marker='o', edgecolors='black', label='Defective')
# Plot threshold line, defined as 3 times the standard deviation of
# reconstruction error on non-defective images
plt.axhline(y=(threshold), color='r', linestyle='-')
plt.legend(loc='upper left')
plt.title('Model ' + model.name + " " + image_type)
plt.ylabel('Anomaly Score')
plt.xlabel('Index')
plt.savefig('Graphs/VAEScores/' + str(dimensions) + model.name + "_" + str(image_type)
            + '_e' + str(epochs) + '_b' + str(batch_size) + '_AS.png', bbox_inches="tight")
plt.close('all')


# Save several image reconstructions to gauge performance of G
ok_idx = [2, 15, 44, 56, 30, 84, 101]
for i in ok_idx:
    img = test_input[i].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)
    reconstructed_img = model.vae.predict(img).reshape(IMG_WIDTH, IMG_HEIGHT)

    im = Image.fromarray(reconstructed_img * 255.0)
    im = im.convert("L")
    im.save('Graphs/VAEReco/' + str(dimensions) + model.name +  "_" + str(i) + "_"
            + str(image_type) + '_e' + str(epochs) + '_b' + str(batch_size)
            + '.png', bbox_inches="tight")

an_idx = [2, 10, 14, 17, 20, 12, 7]
for i in an_idx:
    img = anomalous_input[i].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)
    reconstructed_img = model.vae.predict(img).reshape(IMG_WIDTH, IMG_HEIGHT)

    im = Image.fromarray(reconstructed_img * 255.0)
    im = im.convert("L")
    im.save('Graphs/VAEReco/' + str(dimensions) + model.name + "_" + "anomalous_"
            + str(i) + "_" + str(image_type) + '_e' + str(epochs) + '_b' + str(batch_size)
            + '.png', bbox_inches="tight")


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
