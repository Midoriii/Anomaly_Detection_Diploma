'''
The purpose of this script is to provide some sort of visualisation of what kinds
of images are misclassified when using the embedding + OC-SVM approach.

Models and their best params are hard-coded because frankly it'd be far too much
work for little award to make it any other way.

Data and Models are first loaded, then features are extracted from the data
using the model's predict function. OC-SVM is trained using the given best params
and it's predictions are then used to plot misclassified images (duh).


func: extract_features(images, model): Extracts and returns features from given
images using the predict function of the given model.

func: run_oc_svm(ok_features, faulty_features, testing_ok_features, nu, gamma):
Creates OC-SVM classifier with given nu and gamma, trains it on ok_features and
tests on faulty and ok_extra, returns three lists of predictions.

func: model_eval(ok_data, ok_data_extra, faulty_data, model, nu, gamma):
Uses other methods to perform thorough model evaluation when used with OC-SVM.

func: plot_wrong_predictions(data, predictions, target, title): Plots misclassified
images.
'''
import numpy as np
from cv2 import cv2

from PIL import Image
from sklearn.svm import OneClassSVM
from keras.models import load_model

# Constants
IMG_WIDTH = 768
IMG_HEIGHT = 768


def main():
    '''
    Loads the images and models and performs OC-SVM predictions. Misclassified
    images are plotted.
    '''
    # Load BSE data (and low dim variant)
    bse_ok = np.load("Data/BSE_ok.npy")
    bse_ok_ld = np.load("Data/low_dim_BSE_ok.npy")
    bse_ok_extra = np.load("Data/BSE_ok_extra.npy")
    bse_ok_extra_ld = np.load("Data/low_dim_BSE_ok_extra.npy")
    bse_faulty = np.load("Data/BSE_faulty_extended.npy")
    bse_faulty_ld = np.load("Data/low_dim_BSE_faulty_extended.npy")
    # Load SE data (and low dim variant)
    se_ok = np.load("Data/SE_ok.npy")
    se_ok_ld = np.load("Data/low_dim_SE_ok.npy")
    se_ok_extra = np.load("Data/SE_ok_extra.npy")
    se_ok_extra_ld = np.load("Data/low_dim_SE_ok_extra.npy")
    se_faulty = np.load("Data/SE_faulty_extended.npy")
    se_faulty_ld = np.load("Data/low_dim_SE_faulty_extended.npy")

    # Load desired models
    ae_model_bse = load_model("Model_Saves/Detailed/OcSvm/encoder_extended_TransposeConvAutoencoder_e400_b4_detailed", compile=False)
    ae_model_se = load_model("Model_Saves/Detailed/OcSvm/encoder_extended_HighStrideAutoencoderDeeper_e400_b4_detailed", compile=False)
    siam_model_bse = load_model("Model_Saves/Detailed/OcSvm/embedding_SiameseNetLiteMultipleConvAltTwo_BSE_extended_e40_b4_detailed", compile=False)
    low_dim_siam_model_se = load_model("Model_Saves/Detailed/OcSvm/embedding_low_dims_SiameseNetLiteMultipleConvWithoutDropout_SE_extended_e40_b4_detailed", compile=False)
    # Their best nu values
    nu_values = [0.01, 0.02, 0.01, 0.01]
    # Their best gammas
    gamma_values = ['scale', 'scale', 'auto', 'auto']

    # Eval each model with its attributes .. I wish I knew how to make this
    # in a for loop, but alas the difference in accepted data makes it not worth
    # the effort when this is also decently readable and extendable.

    # First Autoencoders on both BSE and SE data
    print("Transpose AE BSE:")
    model_eval(bse_ok, bse_ok_extra, bse_faulty, ae_model_bse,
               nu_values[0], gamma_values[0])
    print("High Stride AE SE:")
    model_eval(se_ok, se_ok_extra, se_faulty, ae_model_se,
               nu_values[1], gamma_values[1])

    # Then Siamese nets
    print("Siam BSE:")
    model_eval(bse_ok, bse_ok_extra, bse_faulty, siam_model_bse,
               nu_values[2], gamma_values[2])

    # Finally low dim models
    globals_list = globals()
    globals_list['IMG_WIDTH'] = 384
    globals_list['IMG_HEIGHT'] = 384

    print("Low Dim Siam SE:")
    model_eval(se_ok_ld, se_ok_extra_ld, se_faulty_ld, low_dim_siam_model_se,
               nu_values[3], gamma_values[3])



def extract_features(images, model):
    '''
    Extracts features from given image data using model.predict() with a loaded
    pretrained model.

    Arguments:
        images: A numpy array of float32 [0,1] values representing images.
        model: An instantiated pretrained embedding or encoding model.

    Returns:
        A 2D numpy array of extracted features for each input image.
    '''
    images_features = []

    # Get encodings of all given images, using given model.predict()
    for i in range(0, images.shape[0]):
        images_features.append(model.predict(images[i].reshape(1, IMG_WIDTH,
                                                               IMG_HEIGHT, 1)))
    # Convert to numpy array
    images_features = np.asarray(images_features)
    # Reshape to 2D for OC-SVM, -1 means 'make it fit'
    return images_features.reshape(images_features.shape[0], -1)


def run_oc_svm(ok_data_features, faulty_data_features, testing_ok_data_features,
               nu_val=0.02, gamma_val='auto'):
    '''
    Creates and trains an OC-SVM model on given OK features. Predictions are then
    made on extra Ok and Faulty data.

    Arguments:
        ok_data_features: A numpy array representing extracted features from OK data.
            Used for training.
        faulty_data: A numpy array representing extracted features from Faulty data.
            Used for testing.
        testing_ok_data: A numpy array representing extracted features from extra OK data.
            Used for testing.
        nu_val: OC-SVM parameter, refer to:
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
        gamma_val: OC-SVM parameter, refer to the same link as above.

    Returns:
        predictions: A list containing three lists of predictions- OK, OK extra, Faulty.
    '''
    # Create the OC-SVM model
    oc_svm_model = OneClassSVM(gamma=gamma_val, nu=nu_val)
    # Train OC-SVM on OK image features
    oc_svm_model.fit(ok_data_features)

    # Predict on training OK data, testing OK data and Faulty data
    # 1 for inliers, -1 for outliers
    ok_predictions = oc_svm_model.predict(ok_data_features)
    testing_ok_predictions = oc_svm_model.predict(testing_ok_data_features)
    faulty_predictions = oc_svm_model.predict(faulty_data_features)

    print("Training FP:" + str((ok_predictions == -1).sum()))
    print("FP:" + str((testing_ok_predictions == -1).sum()))
    print("FN:" + str((faulty_predictions == 1).sum()))

    return [ok_predictions, testing_ok_predictions, faulty_predictions]


def model_eval(ok_data, ok_data_extra, faulty_data, model, nu_value, gamma_value):
    '''
    Helper method for evaluating models. Calls the other methods to extract
    features, train OC-SVM and plot misclassified images.

    Arguments:
        ok_data: A numpy array of float32 [0,1] values representing training OK images.
        ok_data_extra: A numpy array of float32 [0,1] values representing testing
        OK images.
        faulty_data: A numpy array of float32 [0,1] values representing testing
        Faulty images.
        model: An instantiated pretrained embedding or encoding model.
        nu_value: A parameter for OC-SVM classifier.
        gamma_value: A parameter for OC-SVM classifier.
    '''
    # Extract features using given model
    ok_data_features = extract_features(ok_data, model)
    faulty_data_features = extract_features(faulty_data, model)
    ok_data_extra_features = extract_features(ok_data_extra, model)
    # Train and test OC-SVM on given gamma and nu values
    predictions = run_oc_svm(ok_data_features, faulty_data_features,
                             ok_data_extra_features, nu_val=nu_value,
                             gamma_val=gamma_value)
    # Go through each predictions list and plot wrongly classified images.
    plot_wrong_predictions(ok_data, predictions[0], 1, "OK, FP")
    plot_wrong_predictions(ok_data_extra, predictions[1], 1, "Extra OK, FP")
    plot_wrong_predictions(faulty_data, predictions[2], -1, "Faulty, FN")


def plot_wrong_predictions(data, predictions, target, title):
    '''
    Plots images that have been misclassified. Images to be plotted are given
    by 'data' param, predictions list by 'predictions' and the target the predictions
    should match to not be plotted as faulty by 'target'.

    Arguments:
        data: A numpy array of float32 [0,1] values representing images.
        predictions: A list of predictions.
        target: Target label that the prediction should match.
        title: A string serving as a title of the plot.
    '''
    # Go through each image
    for i in range(0, data.shape[0]):
        # If the prediction doesn't match the target
        if predictions[i] != target:
            # Image needs to be reshaped to 2D first
            img = data[i].reshape(IMG_WIDTH, IMG_HEIGHT)
            # Then it needs to be stretched back into [0,255] values
            img = Image.fromarray(img * 255.0)
            # Converted to grayscale
            img = img.convert('L')
            # And finally shown
            cv2.imshow(title, np.array(img))
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
