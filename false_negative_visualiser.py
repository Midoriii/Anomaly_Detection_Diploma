import numpy as np
import cv2
from PIL import Image
import sys

import matplotlib.pyplot as plt
from keras.models import load_model
from reshape_util import reshape_normalize



img_width = 768
img_height = 768

#Load the saved model itself
model = load_model('Model_Saves/Detailed/BasicAutoencoderEvenDeeperExtraLLR_e600_b4_detailed')
#model = load_model('Model_Saves/Detailed/filtered_BasicAutoencoderEvenDeeper_e50_b4_detailed')

model.summary()

#Load non-anomalous reconstruction errors to get their standard deviation
ok_reconstruction_errors = np.load('Reconstructed/Error_Arrays/BasicAutoencoderEvenDeeperExtraLLR_e600_b4_ROK.npy')
#ok_reconstruction_errors = np.load('Reconstructed/Error_Arrays/filtered_BasicAutoencoderEvenDeeper_e50_b4_ROK.npy')

#Load the anomalous images
anomalies = np.load("Data/Faulty_extended.npy")
print(anomalies.shape)
#Reshape into desired shape for the network
anomalous_input = reshape_normalize(anomalies, img_width, img_height)


#Define the threshold for a picture to be called an anomaly
#to be 3 * the standard deviation of reconstruction error on the OK pics
#threshold = 2.75* np.std(ok_reconstruction_errors)
threshold = 3 * np.std(ok_reconstruction_errors)

#For every anomalous image, encode and decode it, get the reconstruction error and
#compare with threshold - if lower, show the image, it's a false negative
for i in range(0, anomalous_input.shape[0]):
    print(i)
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(anomalous_input[i].reshape(1, img_width, img_height, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(img_width, img_height)
    # Reshape also the original image to compute reconstruction error
    original_image = anomalous_input[i].reshape(img_width, img_height)
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

print("That's all")
