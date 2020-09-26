'''
A script for loading previously saved siamese network and performing
anomaly prediction with it. Prediction is done using prototypes, based on
the same principle as in siamese_network_tester.py.

Models and image dimensions can be manually altered by commenting / uncommenting.

Script outputs results to command line, it also shows the False Negatives
and False Positives, along with undecided images.

func: get_predictions(data, prototypes, model, faulty, image_type):
        Method that performs the prediction itself. Given data, prototypes and model,
        it compares each image with each of the 5 prototypes by using model.predict().
        Afterwards for each image a verdict is printed in the command line. FP, FN
        and undecided images are also plotted.

func: show_misclassified_image(img, msg): Helper function that shows misclassified
        image and waits for a key press to continue.

func: main(): Loads data, prototypes and model. Then calls get_predictions().
        Does so separately for each image type; BSE and SE.
'''
import numpy as np

from cv2 import cv2

from PIL import Image
from keras.models import load_model

# Constants
#IMG_WIDTH = 768
#IMG_HEIGHT = 768
# Low Dim variant
IMG_WIDTH = 384
IMG_HEIGHT = 384

# Gets predictions for given data with given prototypes
# Faulty and Img_Type denote the correctness of Data and type of Images
def get_predictions(data, data_prototypes, model, faulty="OK", img_type="BSE"):
    '''
    Method that does the prediction. Each image from data is compared with all
    given prototypes and anomaly score is computed. Similar images should score > 0.5
    and dissimilar should score <= 0.5. Model.predict() result is rounded and added
    to the image's score. Score of 5 means the tested image is OK, score of 0
    means an anomaly, anything in between requires further assistance, but these
    thresholds are arbitrary and can be customised. Verdict on each image is printed
    in the command line along with the actual state of the image (OK / Faulty).
    False Positives/Negatives along with 'Don't know' images are also plotted.

    Arguments:
        data: Numpy array of float32 [0,1] images to be tested.
        data_prototypes: Numpy array of float32 [0,1] images representing the prototypes.
        model: Loaded trained Siamese Network model that performs predictions.
        faulty: String that helps with distinction of what kind of data was passed,
        either OK or Faulty. Used in the print output as the true label.
        img_type: String that helps with distinction of the image type of data,
        either BSE or SE.
    '''
    for i in range(0, data.shape[0]):
        score = 0
        verdict = ""
        # Run through the prototypes
        for j in range(0, data_prototypes.shape[0]):
            score += np.around(model.predict([
                data[i].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                data_prototypes[j].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)]))

        if score == 5:
            verdict = "OK"
            if faulty == "Faulty":
                show_misclassified_image(data[i], "False Negative")
        elif score == 0:
            verdict = "Faulty"
            if faulty == "OK":
                show_misclassified_image(data[i], "False Positive")
        else:
            verdict = "Don't know"
            if faulty == "OK":
                show_misclassified_image(data[i], "Don't know, OK image")
            else:
                show_misclassified_image(data[i], "Don't know, Faulty image")

        print(img_type + " " + faulty + " clonka #" + str(i) + ": " + verdict)


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


def main():
    '''
    Loads image data and prototypes, divided by the image type; SE vs BSE. Then
    loads a siamese net model (hardcoded for simplicity). Finally for each image type,
    get_predictions() is called on OK data and then on Faulty data.
    '''
    # Loading BSE data
    data_ok = np.load("Data/BSE_ok.npy")
    # Loading the extra OK data for testing purposes
    data_ok_extra = np.load("Data/BSE_ok_extra.npy")
    # Concat them both
    data_ok = np.concatenate((data_ok, data_ok_extra))
    data_faulty = np.load("Data/BSE_faulty_extended.npy")
    data_prototypes = np.load("Data/BSE_prototypes.npy")

    # Loading best BSE Model
    # This one leaves 3 faulty as undecided
    #model = load_model("Model_Saves/Detailed/BasicSiameseNetLowerDropout_BSE_extended_e60_b4_detailed", compile=False)
    # This one leaves 3 OK as undecided
    #model = load_model("Model_Saves/Detailed/SiameseNetLiteMultipleConvAltTwo_BSE_extended_e40_b4_detailed", compile=False)
    # Best lowDim model - and overall best BSE
    model = load_model("Model_Saves/Detailed/low_dims_SiameseNetLiteMultipleConvWithoutDropout_BSE_extended_e40_b4_detailed", compile=False)

    # First get the predictions for BSE OK and then Faulty images
    get_predictions(data_ok, data_prototypes, model, "OK", "BSE")
    get_predictions(data_faulty, data_prototypes, model, "Faulty", "BSE")

    # Loading SE data
    data_ok = np.load("Data/SE_ok.npy")
    # Loading the extra OK data for testing purposes
    data_ok_extra = np.load("Data/SE_ok_extra.npy")
    # Concat them both
    data_ok = np.concatenate((data_ok, data_ok_extra))
    data_faulty = np.load("Data/SE_faulty_extended.npy")
    data_prototypes = np.load("Data/SE_prototypes.npy")
    # Loading best 768x768 SE model
    #model = load_model("Model_Saves/Detailed/SiameseNetLiteMultipleConv_SE_extended_e40_b4_detailed", compile=False)
    # Loading the overall best SE model, low dim 384x384 one
    model = load_model("Model_Saves/Detailed/low_dims_SiameseNetLiteMultipleConvWithoutDropout_SE_extended_e40_b4_detailed", compile=False)

    # Then the same for SE images
    get_predictions(data_ok, data_prototypes, model, "OK", "SE")
    get_predictions(data_faulty, data_prototypes, model, "Faulty", "SE")


if __name__ == "__main__":
    main()
