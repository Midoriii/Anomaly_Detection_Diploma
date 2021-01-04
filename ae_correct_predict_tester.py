'''
Copyright (c) 2021, Štěpán Beneš


This a simple visualiser of images that loaded the Autoencoder model deemed falsely
positive, meaning OK images falsely flagged as faulty.

For simplicity, the model and its saved reconstruction error array are hard-coded.

The threshold for flagging an image as defective is 3 times the standard deviation
of reconstruction errors on OK images, which is a typical threshold for anomaly
detection.

Upon showing an image that was flagged as anomaly (and its reconstruction),
any key press continues the script. The total amount of false positives is also
written to the command line after the script finishes.
'''
from timeit import default_timer as timer
import numpy as np

from cv2 import cv2

from PIL import Image
from keras.models import load_model


IMG_WIDTH = 768
IMG_HEIGHT = 768


# Load the saved model itself
#model = load_model('Model_Saves/Detailed/BasicAutoencoderEvenDeeperExtraLLR_e600_b4_detailed', compile=False)
model = load_model('Model_Saves/Detailed/extended_BasicAutoencoderEvenDeeperExtraLLR_e600_b4_detailed', compile=False)
#model = load_model('Model_Saves/Detailed/filtered_BasicAutoencoderEvenDeeper_e50_b4_detailed', compile=False)

model.summary()

# Load non-anomalous reconstruction errors to get their standard deviation
#ok_reconstruction_errors = np.load('Reconstructed/Error_Arrays/BasicAutoencoderEvenDeeperExtraLLR_e600_b4_ROK.npy')
ok_reconstruction_errors = np.load('Reconstructed/Error_Arrays/extended_BasicAutoencoderEvenDeeperExtraLLR_e600_b4_ROK.npy')
#ok_reconstruction_errors = np.load('Reconstructed/Error_Arrays/filtered_BasicAutoencoderEvenDeeper_e50_b4_ROK.npy')

# Load the OK images
valid_input = np.load("Data/OK.npy")
# Load also the extra OK images, given later on, to test on
valid_input_extra = np.load("Data/OK_extra.npy")
# Concat them both
valid_input = np.concatenate((valid_input, valid_input_extra))
print(valid_input.shape)

# Define the threshold for a picture to be called an anomaly
#threshold = 2.75* np.std(ok_reconstruction_errors)
threshold = 3 * np.std(ok_reconstruction_errors)

# A counter of false positives
falsely_accused = 0

# For every OK image, encode and decode it, get the reconstruction error and
# compare it with threshold - if higher, show the image, it's a false positive
for i in range(0, valid_input.shape[0]):
    print(i)
    start = timer()
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(valid_input[i].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(IMG_WIDTH, IMG_HEIGHT)
    # Reshape also the original image to compute reconstruction error
    original_image = valid_input[i].reshape(IMG_WIDTH, IMG_HEIGHT)
    # Compute the MSE between original and reconstructed
    reconstruction_error = np.square(np.subtract(original_image, reconstructed_img)).mean()

    end = timer()
    print("Prediction Time: " + str(end - start))
    # If the reconstruction error is above threshold, show the image
    if threshold < reconstruction_error:
        print("Falsely hated OK image!")
        # Array has normalized values - need to multiply them again otherwise we get black picture
        im = Image.fromarray(original_image * 255.0)
        rec_im = Image.fromarray(reconstructed_img * 255.0)
        im = im.convert("L")
        rec_im = rec_im.convert("L")
        # Show the OK image and its reconstruction
        cv2.imshow("OK image - original", np.array(im))
        cv2.imshow("OK image - reconstructed", np.array(rec_im))
        cv2.waitKey(0)
        falsely_accused += 1

print(falsely_accused)
print("That's all")
