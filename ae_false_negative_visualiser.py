'''
This a simple visualiser of images that the loaded Autoencoder model deemed falsely
negative, meaning defective images falsely flagged as OK.

For simplicity, the model and its saved reconstruction error array are hard-coded.

The threshold for flagging an image as defective is 3 times the standard deviation
of reconstruction errors on OK images, which is a typical threshold for anomaly
detection.

Upon showing an image that was falsely flagged as OK (and its reconstruction),
any key press continues the script. The total amount of false negatives is also
written to the command line after the script finishes. 
'''
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

# Load the anomalous images
anomalous_input = np.load("Data/Faulty_extended.npy")
print(anomalous_input.shape)

# Define the threshold for a picture to be called an anomaly
#threshold = 2.75* np.std(ok_reconstruction_errors)
threshold = 3 * np.std(ok_reconstruction_errors)

# Missed anomalies counter
missed_anomalies = 0

# For every anomalous image, encode and decode it, get the reconstruction error and
# compare it with threshold - if lower, show the image, it's a false negative
for i in range(0, anomalous_input.shape[0]):
    print(i)
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(anomalous_input[i].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(IMG_WIDTH, IMG_HEIGHT)
    # Reshape also the original image to compute reconstruction error
    original_image = anomalous_input[i].reshape(IMG_WIDTH, IMG_HEIGHT)
    # Compute the MSE between original and reconstructed
    reconstruction_error = np.square(np.subtract(original_image, reconstructed_img)).mean()

    # If the reconstruction error is below threshold, show the image
    if threshold > reconstruction_error:
        print("Undiscovered Anomaly!")
        # Array has normalized values - need to multiply them again otherwise we get black picture
        im = Image.fromarray(original_image * 255.0)
        rec_im = Image.fromarray(reconstructed_img * 255.0)
        im = im.convert("L")
        rec_im = rec_im.convert("L")
        # Show the anomalous image
        cv2.imshow("Anomaly - original", np.array(im))
        cv2.imshow("Anomaly - reconstructed", np.array(rec_im))
        cv2.waitKey(0)
        missed_anomalies = missed_anomalies + 1

print(missed_anomalies)
print("That's all")
