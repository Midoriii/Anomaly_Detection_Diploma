'''
The purpose of this script is to test One Class SVM model, sometimes used
for anomaly detection purposes.

OC-SVM is trained on only OK data features and tries to use a hypersphere to
encompass the given instances. Anything otuside of the hypersphere should be
treated as an anomaly.

As OC-SVM requires extracted features as its training/testing input, a suitable
pre-trained encoding (part of Autoencoders) or embedding (branch in Siamese Nets)
model needs to be provided.

After training is done, the OC-SVM is tested on extracted features of OK and Faulty
data. The amount of False Negatives and False Positives is finally printed out.

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
    Loads OK and Faulty data and the encoding / embedding models and calls
    train_oc_svm(), once for each type of data, SE vs BSE.
    '''
    # Load SE and BSE ok and faulty data
    bse_ok_data = np.load("Data/BSE_ok.npy")
    bse_ok_data_extra = np.load("Data/BSE_ok_extra.npy")
    bse_ok_data = np.concatenate((bse_ok_data, bse_ok_data_extra))
    bse_faulty_data = np.load("Data/BSE_faulty_extended.npy")

    se_ok_data = np.load("Data/SE_ok.npy")
    se_ok_data_extra = np.load("Data/SE_ok_extra.npy")
    se_ok_data = np.concatenate((se_ok_data, se_ok_data_extra))
    se_faulty_data = np.load("Data/SE_faulty_extended.npy")

    # Load the Encoding or BSE Embedding model
    model = load_model("Model_Saves/Detailed/encoder_extended_BasicAutoencoderEvenDeeperExtraLLR_e600_b4_detailed", compile=False)
    #
    print("BSE:")
    train_oc_svm(bse_ok_data, bse_faulty_data, model)
    # Load SE embedding model
    #
    print("SE:")
    train_oc_svm(se_ok_data, se_faulty_data, model)


def train_oc_svm(ok_data, faulty_data, model):
    '''
    Extracts the features of ok_data and faulty_data using the .predict() method
    of given embedding or encoding model. One Class SVM is then trained on the
    OK features. Predictions are then made on both Ok and Faulty data. Finally,
    the number of false positives and false negatives is printed out.

    Arguments:
        ok_data: A numpy array of float32 [0,1] values representing images.
        faulty_data: A numpy array of float32 [0,1] values representing images.
        model: An instantiated embedding or encoding model.
    '''
    ok_data_features = []
    faulty_data_features = []
    # Create the OC-SVM model
    oc_svm_model = OneClassSVM(gamma='auto')
    # Get encodings of all OK and Faulty images
    for i in range(0, ok_data.shape[0]):
        ok_data_features.append(model.predict(ok_data[i].reshape(1, IMG_WIDTH,
                                                                 IMG_HEIGHT, 1)))
    for i in range(0, faulty_data.shape[0]):
        faulty_data_features.append(model.predict(faulty_data[i].reshape(1, IMG_WIDTH,
                                                                         IMG_HEIGHT, 1)))
    # Train OC-SVM on OK images
    oc_svm_model.fit(ok_data_features)
    # Predict on faulty data and also ok data ?
    # 1 for inliers, -1 for outliers
    ok_predictions = oc_svm_model.predict(ok_data_features)
    faulty_predictions = oc_svm_model.predict(faulty_data_features)

    print("FP:")
    print(ok_predictions.count(-1))
    print("FN:")
    print(faulty_predictions.count(1))



if __name__ == "__main__":
    main()
