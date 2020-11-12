'''
bla
'''
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from Models.biGAN.BasicBigan import BasicBigan
from Models.biGAN.BasicBiganTTUR import BasicBiganTTUR
from Models.biGAN.BasicBiganHF import BasicBiganHF
from Models.biGAN.BasicBiganAlt import BasicBiganAlt
from Models.biGAN.BasicBiganAVGPool import BasicBiganAVGPool
from Models.biGAN.BasicBiganWoutBN import BasicBiganWoutBN
from Models.biGAN.BasicBiganXEntropy import BasicBiganXEntropy
from Models.biGAN.BasicBiganXEntropyShallower import BasicBiganXEntropyShallower
from Models.biGAN.BasicBiganXEntropyTTUR import BasicBiganXEntropyTTUR
from Models.biGAN.BasicBiganXEntropyHLR import BasicBiganXEntropyHLR
from Models.biGAN.BasicBiganXEntropyExtraHLR import BasicBiganXEntropyExtraHLR
from Models.biGAN.BasicBiganXEntropyLLR import BasicBiganXEntropyLLR
from Models.biGAN.BasicBiganXEntropyExtraBN import BasicBiganXEntropyExtraBN
from Models.biGAN.BasicBiganXEntropyExtraGencTraining import BasicBiganXEntropyExtraGencTraining
from Models.biGAN.BasicBiganHLR import BasicBiganHLR
from Models.biGAN.BasicBiganExtraHLR import BasicBiganExtraHLR
from Models.biGAN.BasicBiganLLR import BasicBiganLLR
from Models.biGAN.BasicBiganExtraBN import BasicBiganExtraBN
from Models.biGAN.BasicBiganExtraBNHigherWeightClip import BasicBiganExtraBNHigherWeightClip
from Models.biGAN.BasicBiganExtraGencTraining import BasicBiganExtraGencTraining
from Models.biGAN.BasicBiganShallower import BasicBiganShallower
from Models.biGAN.BasicBiganHiDropout import BasicBiganHiDropout
from Models.biGAN.BasicBiganExtraDropout import BasicBiganExtraDropout
from Models.biGAN.BasicBiganLowerWeightClip import BasicBiganLowerWeightClip
from Models.biGAN.BasicBiganHigherWeightClip import BasicBiganHigherWeightClip
from Models.biGAN.BasicBiganMixedLoss import BasicBiganMixedLoss
from Models.biGAN.lowerDimBigan import lowerDimBigan
from Models.biGAN.lowerDimBiganExtraBN import lowerDimBiganExtraBN
from Models.biGAN.lowerDimBiganHigherWeightClip import lowerDimBiganHigherWeightClip
from Models.biGAN.lowerDimBiganXEntropy import lowerDimBiganXEntropy

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
short_options = "e:b:m:t:d:"
long_options = ["epochs=", "batch_size=", "model=", "type=", "dimensions="]
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
    elif current_argument in ("-d", "--dimensions"):
        dimensions = current_value


# Load appropriate data, selected by image type
if image_type == "BSE":
    if dimensions == "low_dim_":
        train_input = np.load("DataBigan/low_dim_BSE_ok_bigan.npy")
        test_input = np.load("DataBigan/low_dim_BSE_ok_extra_bigan.npy")
        test_input = np.concatenate((train_input, test_input))
        anomalous_input = np.load("DataBigan/low_dim_BSE_faulty_extended_bigan.npy")
    else:
        train_input = np.load("DataBigan/extra_low_dim_BSE_ok_bigan.npy")
        test_input = np.load("DataBigan/extra_low_dim_BSE_ok_extra_bigan.npy")
        test_input = np.concatenate((train_input, test_input))
        anomalous_input = np.load("DataBigan/extra_low_dim_BSE_faulty_extended_bigan.npy")
elif image_type == "SE":
    if dimensions == "low_dim_":
        train_input = np.load("DataBigan/low_dim_SE_ok_bigan.npy")
        test_input = np.load("DataBigan/low_dim_SE_ok_extra_bigan.npy")
        test_input = np.concatenate((train_input, test_input))
        anomalous_input = np.load("DataBigan/low_dim_SE_faulty_extended_bigan.npy")
    else:
        train_input = np.load("DataBigan/extra_low_dim_SE_ok_bigan.npy")
        test_input = np.load("DataBigan/extra_low_dim_SE_ok_extra_bigan.npy")
        test_input = np.concatenate((train_input, test_input))
        anomalous_input = np.load("DataBigan/extra_low_dim_SE_faulty_extended_bigan.npy")
else:
    print("Wrong Image Type specified!")
    sys.exit()


# Choose desired model
if desired_model == "BasicBigan":
    model = BasicBigan(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganTTUR":
    model = BasicBiganTTUR(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganHF":
    model = BasicBiganHF(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganAlt":
    model = BasicBiganAlt(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganAVGPool":
    model = BasicBiganAVGPool(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganWoutBN":
    model = BasicBiganWoutBN(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganXEntropy":
    model = BasicBiganXEntropy(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganXEntropyShallower":
    model = BasicBiganXEntropyShallower(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganXEntropyTTUR":
    model = BasicBiganXEntropyTTUR(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganXEntropyHLR":
    model = BasicBiganXEntropyHLR(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganXEntropyExtraHLR":
    model = BasicBiganXEntropyExtraHLR(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganXEntropyLLR":
    model = BasicBiganXEntropyLLR(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganXEntropyExtraBN":
    model = BasicBiganXEntropyExtraBN(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganXEntropyExtraGencTraining":
    model = BasicBiganXEntropyExtraGencTraining(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganHLR":
    model = BasicBiganHLR(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganExtraHLR":
    model = BasicBiganExtraHLR(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganExtraBN":
    model = BasicBiganExtraBN(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganExtraBNHigherWeightClip":
    model = BasicBiganExtraBNHigherWeightClip(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganExtraGencTraining":
    model = BasicBiganExtraGencTraining(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganLLR":
    model = BasicBiganLLR(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganLowerWeightClip":
    model = BasicBiganLowerWeightClip(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganHigherWeightClip":
    model = BasicBiganHigherWeightClip(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganHiDropout":
    model = BasicBiganHiDropout(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganExtraDropout":
    model = BasicBiganExtraDropout(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganMixedLoss":
    model = BasicBiganMixedLoss(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "BasicBiganShallower":
    model = BasicBiganShallower(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "lowerDimBigan":
    model = lowerDimBigan(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "lowerDimBiganExtraBN":
    model = lowerDimBiganExtraBN(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "lowerDimBiganHigherWeightClip":
    model = lowerDimBiganHigherWeightClip(IMG_WIDTH, batch_size=batch_size)
elif desired_model == "lowerDimBiganXEntropy":
    model = lowerDimBiganXEntropy(IMG_WIDTH, batch_size=batch_size)
else:
    print("Wrong Model specified!")
    sys.exit()


# Train it
model.train(train_input, epochs=epochs)
# Save it
model.save_model(epochs, image_type, dimensions)


# The X axis of the plots - number of epochs .. +1 since it's arange
X = np.arange(1, epochs+1)
# Plot and save G, R, D losses .. use tab: colors
plt.plot(X, model.g_losses, label="Gen loss", color='tab:green')
plt.plot(X, model.e_losses, label="Enc loss", color='tab:blue')
plt.plot(X, model.d_losses, label="Dis loss", color='tab:red')
plt.legend(loc='upper right')
plt.title('Model ' + model.name + " " + image_type)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('Graphs/Losses/' + str(dimensions) + model.name + "_" + str(image_type)
            + '_e' + str(epochs) + '_b' + str(batch_size) + '.png', bbox_inches="tight")
plt.close('all')

# Plot and save Dfake and Dreal accuracies
plt.plot(X, model.dr_acc, label="Real Accuracy", color='tab:green')
plt.plot(X, model.df_acc, label="Fake Accuracy", color='tab:red')
plt.legend(loc='upper right')
plt.title('Model ' + model.name + " " + image_type)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('Graphs/Accuracies/' + str(dimensions) + model.name + "_" + str(image_type)
            + '_e' + str(epochs) + '_b' + str(batch_size) + '.png', bbox_inches="tight")
plt.close('all')


# Calculate Anomaly scores for test input
test_scores = []
for i in range(0, test_input.shape[0]):
    test_scores.append(model.predict(test_input[i]))

# Calculate Anomaly scores for anomalous input
anomalous_scores = []
for i in range(0, anomalous_input.shape[0]):
    anomalous_scores.append(model.predict(anomalous_input[i]))

# Plot and save the anomaly scores
X1 = np.arange(0, len(test_scores))
X2 = np.arange(len(test_scores), len(test_scores) + len(anomalous_scores))

plt.scatter(X1, test_scores, c='g', s=10,
            marker='o', edgecolors='black', label='Without Defect')
plt.scatter(X2, anomalous_scores, c='r', s=10,
            marker='o', edgecolors='black', label='Defective')
plt.legend(loc='upper left')
plt.title('Model ' + model.name + " " + image_type)
plt.ylabel('Anomaly Score')
plt.xlabel('Index')
plt.savefig('Graphs/biGANScores/' + str(dimensions) + model.name + "_" + str(image_type)
            + '_e' + str(epochs) + '_b' + str(batch_size) + '_AS.png', bbox_inches="tight")
plt.close('all')


# Save several image reconstructions to gauge performance of G
ok_idx = [2, 15, 44, 56, 30, 84, 101]
for i in ok_idx:
    img = test_input[i].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)
    reconstructed_img = model.e.predict(img)
    reconstructed_img = model.g.predict(reconstructed_img).reshape(IMG_WIDTH, IMG_HEIGHT)

    im = Image.fromarray((reconstructed_img * 127.5) + 127.5)
    im = im.convert("L")
    im.save('Graphs/biGANReco/' + str(dimensions) + model.name +  "_" + str(i) + "_"
            + str(image_type) + '_e' + str(epochs) + '_b' + str(batch_size)
            + '.png', bbox_inches="tight")

an_idx = [2, 10, 14, 17, 20, 12, 7]
for i in an_idx:
    img = anomalous_input[i].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)
    reconstructed_img = model.e.predict(img)
    reconstructed_img = model.g.predict(reconstructed_img).reshape(IMG_WIDTH, IMG_HEIGHT)

    im = Image.fromarray((reconstructed_img * 127.5) + 127.5)
    im = im.convert("L")
    im.save('Graphs/biGANReco/' + str(dimensions) + model.name + "_" + "anomalous_"
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
