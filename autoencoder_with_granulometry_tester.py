'''

'''
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from cv2 import cv2

from granulo_utils import reshape_img_from_float
from granulo_utils import threshold_image
from granulo_utils import adaptive_threshold_image
from granulo_utils import granulometry_score
from granulo_utils import remove_center


# Constants
IMG_WIDTH = 768
IMG_HEIGHT = 768
BIN_THRESHOLD = 55
GR_THRESHOLD = 50

# This should go to main()
# Load BSE data
ok_data = np.load("Data/BSE_ok.npy")
faulty_data = np.load("Data/BSE_faulty_extended.npy")

# Load Autoencoder model and error array
model = load_model('Model_Saves/Detailed/extended_BasicAutoencoderEvenDeeperExtraLLR_e600_b4_detailed', compile=False)
model.summary()
ok_reconstruction_errors = np.load('Reconstructed/Error_Arrays/extended_BasicAutoencoderEvenDeeperExtraLLR_e600_b4_ROK.npy')

# This should be a method
# For each OK / Faulty image
    ae_score = 0
    gr_score = 0
    # Get reconstruction score as per usual, anything above 3 * std is anomaly
    AE_THRESHOLD = 3 * np.std(ok_reconstruction_errors)

    # Get anomaly scores from granulometry just like in granulometry_tester.py
    # Create structuring element for image opening
    struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

# Plot the final score for each image by index



if __name__ == "__main__":
    main()
