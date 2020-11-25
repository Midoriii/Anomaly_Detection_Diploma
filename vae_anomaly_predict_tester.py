'''
A script for loading previously trained VAE models and performing anomaly detection
with them. Prediction is done using reconstruction error and a threshold for it,
calculated on training OK data.

Since only lower dimension (384x384) data has been used with VAE, the same
happens here.

Script outputs results to command line (stdout) and shows misclassified images.


func: calculate_threshold(model, data, coef):
        Method that calculates threshold for anomaly detection.

func: predict(model, orig_img):
        Method that performs prediction (reconstruction) on a single image.

func: show_misclassified_image(img, msg):
        Method that shows the misclassified image.

func: get_predictions(data, threshold, model, faulty, image_type):
        Method that performs the predictions on all images by calling predict().
        Decides the outcome for each image and prints/shows the results.

func: main():
        Loads images and models and calls the other methods. Separately done
        for each image type BSE vs SE.
'''
import numpy as np

from cv2 import cv2

from PIL import Image
from keras.models import load_model

# Constants
IMG_WIDTH = 384
IMG_HEIGHT = 384


def calculate_threshold(model, data, coef):
    '''
    bla

    Arguments:
        model: A pretrained VAE model capable of prediction.
        data: A numpy array of float32 [0,1] arrays representing images.
        coef: A multiplier of the threshold value.

    Returns:
        threshold: A calculated threshold value.
    '''
    scores = []
    for i in range(0, data.shape[0]):
            scores.append(model.predict(data[i]))
    return coef * np.std(scores)


def predict(model, orig_img):
    '''
    Reconstructs given orig_img using model and calculated reconstrction error
    as MSE between orig_img and its reconstruction.

    Arguments:
        model: A pretrained VAE model capable of prediction.
        orig_img: A numpy array of float32 [0,1] representing an image.

    Returns:
        reconstruction_error: A value representing reco. error.
    '''
    orig_img = orig_img.reshape(IMG_WIDTH, IMG_HEIGHT)
    # Reshape into valid input
    img = orig_img.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)
    # Get the reconstruction
    re_img = model.predict(img).reshape(IMG_WIDTH, IMG_HEIGHT)
    # Calculate reco error
    reconstruction_error = np.square(np.subtract(orig_img, re_img)).mean()
    return reconstruction_error


def show_misclassified_image(img, msg):
    '''
    Reshapes, stretches and finally shows given image.

    Arguments:
        img: A numpy array of float32 [0,1] representing an image.
        msg: Title to display when showing the image.
    '''
    # Image needs to be reshaped to 2D first
    img = img.reshape(IMG_WIDTH, IMG_HEIGHT)
    # Then it needs to be stretched back into [0,255] values
    img = Image.fromarray(img * 255.0)
    # Converted to grayscale
    img = img.convert('L')
    # And finally shown
    cv2.imshow(msg, np.array(img))
    cv2.waitKey(0)


def get_predictions(data, threshold, model, faulty="OK", img_type="BSE"):
    '''
    A method that performs prediction using given model on all given data.
    For each image, an anomaly score is computed, and any score above threshold
    is flagged as an anomaly, and any below as an OK image. Results for each image
    are written to standard output, misclassified images are shown.

    Arguments:
        data: A numpy array of [0,1] images.
        threshold: A value for anomaly detection - any score above is an anomaly.
        model: An instantiated trained VAE model capable od prediction.
        faulty: String that helps with distinction of what kind of data was passed,
        either OK or Faulty. Used in the print output as the true label.
        img_type: String that helps with distinction of the image type of data,
        either BSE or SE.
    '''
    for i in range(0, data.shape[0]):
        score = model.predict(data[i])

        if faulty == "OK":
            if score >= threshold:
                verdict = "Defective"
                show_misclassified_image(data[i], "False Positive")
            else:
                verdict = "OK"
        else:
            if score < threshold:
                verdict = "OK"
                show_misclassified_image(data[i], "False Negative")
            else:
                verdict = "Defective"

        print(img_type + " " + faulty + " clonka #" + str(i) + ": " + verdict)


def main():
    '''
    Loads image data and prototypes, divided by the image type; SE vs BSE. Then
    loads a VAE model (hardcoded for simplicity). Afterwards, calculates the threshold
    for anomaly detection on training OK data. Finally for each image type,
    get_predictions() is called on OK data and then on Faulty data.
    '''
    # Loading BSE data
    data_ok = np.load("Data/low_dim_BSE_ok.npy")
    data_ok_extra = np.load("Data/low_dim_BSE_ok_extra.npy")
    data_faulty = np.load("Data/low_dim_BSE_faulty_extended.npy")
    # Concat the ok data
    data_ok_all = np.concatenate((data_ok, data_ok_extra))

    # Loading best BSE Model
    model = load_model("Model_Saves/Detailed/vae_low_dim_BasicVAE_HiRLFactor_LowLatDimBSE_e1200_detailed", compile=False)
    # Get the BSE threshold
    threshold = calculate_threshold(model, data_ok, 3)
    # First get the predictions for BSE OK and then Faulty images
    get_predictions(data_ok_all, threshold, model, "OK", "BSE")
    get_predictions(data_faulty, threshold, model, "Defective", "BSE")


    # Loading SE data
    data_ok = np.load("Data/low_dim_SE_ok.npy")
    data_ok_extra = np.load("Data/low_dim_SE_ok_extra.npy")
    data_faulty = np.load("Data/low_dim_SE_faulty_extended.npy")
    # Concat the ok data
    data_ok_all = np.concatenate((data_ok, data_ok_extra))

    # Loading best SE Model
    model = load_model("Model_Saves/Detailed/vae_low_dim_BasicVAE_HiRLFactor_LowLatDimSE_e1200_detailed", compile=False)
    # Get the SE threshold
    threshold = calculate_threshold(model, data_ok, 4)
    # Then the same for SE images
    get_predictions(data_ok_all, threshold, model, "OK", "SE")
    get_predictions(data_faulty, threshold, model, "Defective", "SE")


if __name__ == "__main__":
    main()
