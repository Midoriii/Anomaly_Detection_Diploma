'''
The purpose of this script is to test One Class SVM model, sometimes used
for anomaly detection purposes.

OC-SVM is trained on only OK data features and tries to use a hypersphere to
encompass the given instances. Anything otuside of the hypersphere should be
treated as an anomaly.

As OC-SVM requires extracted features as its training/testing input, a suitable
pre-trained encoding (part of Autoencoders) or embedding (branch in Siamese Nets)
model needs to be provided.

After training is done, the OC-SVM is tested on extracted features of extra OK and
Faulty data. The amount of False Negatives and False Positives is finally printed out.

Results will be added here.
'''
import numpy as np

from sklearn.svm import OneClassSVM
from keras.models import load_model


# Constants
IMG_WIDTH = 768
IMG_HEIGHT = 768


def main():
    '''
    Loads OK, Faulty and extra OK data and the encoding / embedding models, calls
    extract_features() to get the features of the loaded data, and finally calls
    train_oc_svm(), once for each type of data, SE vs BSE.
    '''
    # Load SE and BSE ok and faulty data
    bse_ok_data = np.load("Data/BSE_ok.npy")
    bse_ok_data_extra = np.load("Data/BSE_ok_extra.npy")
    bse_faulty_data = np.load("Data/BSE_faulty_extended.npy")

    se_ok_data = np.load("Data/SE_ok.npy")
    se_ok_data_extra = np.load("Data/SE_ok_extra.npy")
    se_faulty_data = np.load("Data/SE_faulty_extended.npy")

    # Load the Encoding or BSE Embedding model
    #model = load_model("Model_Saves/Detailed/encoder_extended_BasicAutoencoderEvenDeeperExtraLLR_e600_b4_detailed", compile=False)
    model = load_model("Model_Saves/Detailed/embedding_BasicSiameseNetLowerDropout_BSE_extended_e60_b4_detailed", compile=False)
    print("BSE:")
    # Extract features from each type of data
    bse_ok_data_features = extract_features(bse_ok_data, model)
    bse_faulty_data_features = extract_features(bse_faulty_data, model)
    bse_ok_data_extra_features = extract_features(bse_ok_data_extra, model)
    # Train the model and get the results
    train_oc_svm(bse_ok_data_features, bse_faulty_data_features,
                 bse_ok_data_extra_features, model)

    # Load SE embedding model
    model = load_model("Model_Saves/Detailed/embedding_BasicSiameseNetWithoutDropout_SE_extended_e40_b4_detailed", compile=False)
    print("SE:")
    se_ok_data_features = extract_features(se_ok_data, model)
    se_faulty_data_features = extract_features(se_faulty_data, model)
    se_ok_data_extra_features = extract_features(se_ok_data_extra, model)
    train_oc_svm(se_ok_data_features, se_faulty_data_features,
                 se_ok_data_extra_features, model)


def train_oc_svm(ok_data_features, faulty_data_features,
                 testing_ok_data_features, model, nu_val=0.02):
    '''
    Creates and trains an OC-SVM model on given OK features. Predictions are then
    made on extra Ok and Faulty data. Finally, the number of false positives and
    false negatives is printed out.

    Arguments:
        ok_data_features: A numpy array representing extracted features from OK data.
            Used for training.
        faulty_data: A numpy array representing extracted features from Faulty data.
            Used for testing.
        testing_ok_data: A numpy array representing extracted features from extra OK data.
            Used for testing.
        model: An instantiated pretrained embedding or encoding model.
        nu_val: OC-SVM parameter, refer to:
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    '''
    # Create the OC-SVM model, gamma='auto' vs 'scale', 'auto' seems better
    oc_svm_model = OneClassSVM(gamma='auto', nu=nu_val)
    # Train OC-SVM on OK image features
    oc_svm_model.fit(ok_data_features)

    # Predict on training OK data and Faulty data
    # 1 for inliers, -1 for outliers
    testing_ok_predictions = oc_svm_model.predict(testing_ok_data_features)
    faulty_predictions = oc_svm_model.predict(faulty_data_features)

    print("FP:")
    print((testing_ok_predictions == -1).sum())
    print("FN:")
    print((faulty_predictions == 1).sum())


def extract_features(images, model):
    '''
    Extracts features from given image data using model.predict() of a loaded
    pretrained model.

    Arguments:
        images: A numpy array of float32 [0,1] values representing images.
        model: An instantiated pretrained embedding or encoding model.

    Returns:
        A numpy array of extracted features for each input image.
    '''
    images_features = []

    # Get encodings of all given images, using given model.predict()
    for i in range(0, images.shape[0]):
        images_features.append(model.predict(images_data[i].reshape(1, IMG_WIDTH,
                                                                     IMG_HEIGHT, 1)))
    # Convert to numpy array
    images_features = np.asarray(images_features)
    # Reshape to 2D for OC-SVM, -1 means 'make it fit'
    return images_features.reshape(images_features.shape[0], -1)


if __name__ == "__main__":
    main()
