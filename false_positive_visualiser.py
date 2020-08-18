import numpy as np

from cv2 import cv2

from PIL import Image
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

#Load the OK images
part1 = np.load("Data/OK_1.npy")
part2 = np.load("Data/OK_2.npy")
data = np.concatenate((part1, part2))
print(data.shape)
#Reshape into desired shape for the network
valid_input = reshape_normalize(data, img_width, img_height)

#Define the threshold for a picture to be called an anomaly
#to be 3 * the standard deviation of reconstruction error on the OK pics
#threshold = 2.75* np.std(ok_reconstruction_errors)
threshold = 3 * np.std(ok_reconstruction_errors)

falsely_accused = 0

#For every OK image, encode and decode it, get the reconstruction error and
#compare with threshold - if higher, show the image, it's a false positive
for i in range(0, valid_input.shape[0]):
    print(i)
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(valid_input[i].reshape(1, img_width, img_height, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(img_width, img_height)
    # Reshape also the original image to compute reconstruction error
    original_image = valid_input[i].reshape(img_width, img_height)
    # Compute the MSE between original and reconstructed
    reconstruction_error = np.square(np.subtract(original_image, reconstructed_img)).mean()

    # If the reconstruction error is above threshold, show the image
    if threshold < reconstruction_error:
        print("Falsely hated OK image!")
        # Array has normalized values - need to multiply them again otherwise we get black picture
        im = Image.fromarray(original_image * 255.0)
        rec_im = Image.fromarray(reconstructed_img * 255.0)
        im = im.convert("L")
        rec_im = rec_im.convert("L")
        # Show the OK image
        cv2.imshow("OK image - original", np.array(im))
        cv2.imshow("OK image - reconstructed", np.array(rec_im))
        cv2.waitKey(0)
        falsely_accused = falsely_accused + 1

print(falsely_accused)
print("That's all")
