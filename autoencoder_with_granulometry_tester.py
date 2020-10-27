'''
Short script to try granulometry with the best Autoencoder model. Granulometry
params used are the best found so far.

Results are shown on a graph, where each image is assigned a score.
3 - Both Autoencoder and Granulometry mark the image as defective.
2 - Only Granulometry marks the image as defective.
1 - Only Autoencoder marks the image as adefective.
0 - Image marked as OK by both AE and Granulo.
Green dots represent truly OK images, red dots the faulty ones.

It appears that using Granulometry can help, as it marked 3 images as faulty
of the 6 that the best Autoencoder model marked as False Negatives.
'''
from collections import Counter
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from cv2 import cv2

from granulo_utils import perform_binary_granulometry


# Constants
IMG_WIDTH = 768
IMG_HEIGHT = 768
BIN_THRESHOLD = 55
GR_THRESHOLD = 50


def main():
    '''
    The Main function of this script. Loads the BSE data (no use testing on SE),
    loads the model and its reconstruction errors array to set the anomaly threshold,
    calls get_scores() to get results for both OK and faulty BSE images and plots
    the scores for each image.
    '''
    # Load BSE data
    ok_data = np.load("Data/BSE_ok.npy")
    # Load the extra ok images for testing
    ok_data_extra = np.load("Data/BSE_ok_extra.npy")
    # Concat the ok data
    ok_data = np.concatenate((ok_data, ok_data_extra))
    faulty_data = np.load("Data/BSE_faulty_extended.npy")

    # Load Autoencoder model and error array
    model = load_model('Model_Saves/Detailed/extended_BasicAutoencoderEvenDeeperExtraLLR_e600_b4_detailed', compile=False)
    model.summary()
    ok_reconstruction_errors = np.load('Reconstructed/Error_Arrays/extended_BasicAutoencoderEvenDeeperExtraLLR_e600_b4_ROK.npy')
    # Anything above 3 * std is anomaly
    AE_THRESHOLD = 3 * np.std(ok_reconstruction_errors)
    # For each OK / Faulty image
    ok_scores = get_scores(ok_data, model, AE_THRESHOLD)
    faulty_scores = get_scores(faulty_data, model, AE_THRESHOLD)
    # Use Counters and OrderedDicts as in siamese and triplet runner scripts
    ok_counter = Counter(ok_scores)
    faulty_counter = Counter(faulty_scores)

    for i in range(0, 4):
        if not ok_counter.get(i):
            ok_counter[i] = 0
        if not faulty_counter.get(i):
            faulty_counter[i] = 0

    ok_ordered = OrderedDict(sorted(ok_counter.items(), key=lambda x: x[0]))
    faulty_ordered = OrderedDict(sorted(faulty_counter.items(), key=lambda x: x[0]))
    # Plot the final score for each image by index
    # X axis coords representing scores 0-3, which in turn represent which
    # Model called out an anomaly
    X = np.arange(4)

    ok_bars = plt.bar(X + 0.15, ok_ordered.values(), color='tab:green', width=0.30,
                      label="Without Defect")
    faulty_bars = plt.bar(X - 0.15, faulty_ordered.values(), color='tab:red', width=0.30,
                          label="Defective")
    plt.legend(loc='upper center')
    plt.title('AE & Granulo scores')
    plt.xticks(np.arange(0.0, 4.0, 1.0), ('None', 'AE', 'GR', 'Both'))
    plt.xlabel('Flagged as Anomaly by:')
    plt.ylabel('Image count')
    autolabel(ok_bars, 'tab:green')
    autolabel(faulty_bars, 'tab:red')
    plt.show()


def get_scores(images, model, ae_threshold):
    '''
    A method that performs anomaly detection on given images, using given
    Autoencoder model and its detection threshold, and using granulometry
    with params given as constants in this script. Each image is scored
    depending on the results of both anomaly detection methods.

    3 - Both Autoencoder and Granulometry mark the image as defective.
    2 - Only Granulometry marks the image as defective.
    1 - Only Autoencoder marks the image as adefective.
    0 - Image marked as OK by both AE and Granulo.

    Arguments:
        images: A numpy array of float32 [0,1] images.
        model: Instantiated autoencoder model to perform predictions.
        ae_threshold: Threshold for anomaly detection using autoencoder prediction.

    Returns:
        resulting_scores: A list of achieved scores, one value for each image.
    '''
    resulting_scores = []
    # Create structuring element for image opening
    struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    # For each image
    for i in range(0, images.shape[0]):
        ae_result = 0
        gr_result = 0
        # Get reconstruction error as per usual
        # Every image needs to be reshaped into 1,768,768,1
        reconstructed_img = model.predict(images[i].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
        # The reconstructed image afterwards needs to be reshaped back into 768 x 768
        reconstructed_img = reconstructed_img.reshape(IMG_WIDTH, IMG_HEIGHT)
        # Reshape also the original image to compute reconstruction error
        original_image = images[i].reshape(IMG_WIDTH, IMG_HEIGHT)
        # Compute the MSE between original and reconstructed
        reconstruction_error = np.square(np.subtract(original_image, reconstructed_img)).mean()
        # Tag as 1 if the error is above the threshold
        if reconstruction_error > ae_threshold:
            ae_result = 1


        # Get anomaly scores from granulometry just like in granulometry_tester.py
        gr_score = perform_binary_granulometry(images[i], IMG_WIDTH, IMG_HEIGHT,
                                               BIN_THRESHOLD, struct_element)
        # Tag as 1, meaning an anomaly based on granulometry score
        if gr_score > GR_THRESHOLD:
            gr_result = 1


        # Final score decision based on results so far
        # Both methods detected faulty, result = 3
        if gr_result == 1 and ae_result == 1:
            result = 3
        # Only granulometry marked the image as faulty, result = 2
        elif gr_result == 1 and ae_result == 0:
            result = 2
        # Only autoencoder marked the image as faulty, result = 1
        elif gr_result == 0 and ae_result == 1:
            result = 1
        # None marked the image as faulty, result = 0
        else:
            result = 0
        # Append the resulting score
        resulting_scores.append(result)

    return resulting_scores


def autolabel(rects, color):
    """
    A helper function for barplot labeling. I chose to include this with the script
    to prevent additional file importing.

    Arguments:
        rects: A list of rectangles representing bar plots.
        color: Desired color of the labels.
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 0.1 + height,
                 '%d' % int(height), color=color,
                 ha='center', va='bottom')


if __name__ == "__main__":
    main()
