import numpy as np
import sys

from keras.models import load_model


# Just a helper script to see what's going wrong when training the net
img_width = 768
img_height = 768
# Only concerned with BSE data as SE works fine
pairs_left = np.load("DataHuge/BSE_pairs_left.npy")
pairs_right = np.load("DataHuge/BSE_pairs_right.npy")
pairs_labels = np.load("DataHuge/BSE_pairs_labels.npy")

#Load the saved model itself
model = load_model('Model_Saves/Test/SiameseNetDeeper_BSE_e40_b4_detailed')
model.summary()

# Iterate through all the pairs and print results of predict
for i in range (0, pairs_labels.shape[0]):
    prediction = model.predict(pairs_left[i].reshape((1, img_width, img_height, 1)),
                               pairs_right[i].reshape((1, img_width, img_height, 1)))

    print("Predicted: " + str(prediction) + ", Label: " + str(pairs_labels[i]))
