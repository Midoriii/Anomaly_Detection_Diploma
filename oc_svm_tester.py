'''
The purpose of this script is to test One Class SVM model, sometimes used
for anomaly detection purposes.

OC-SVM is trained on only OK data features and tries to use a hypersphere to
encompass the given instances. Anything otuside of the hypersphere should be
treated as an anomaly.

As OC-SVM requires extracted features as its training/testing input, a suitable
pre-trained encoding (part of Autoencoders) or embedding (branch in Siamese/Triplet
Nets) model needs to be provided.

After training is done, the OC-SVM is tested on extracted features of extra OK and
Faulty data. The amount of False Negatives and False Positives is finally printed out.
This is done for several possible values of 'nu' parameter of OC-SVM models, to try
and find the best model and config.
'''
import numpy as np

from sklearn.svm import OneClassSVM
from keras.models import load_model


# Constants
IMG_WIDTH = 768
IMG_HEIGHT = 768


def main():
    '''
    Loads OK, Faulty and extra OK data and the Encoding / Embedding models. For
    each Encoding and Embedding model the data features are extracted using
    extract_feature() method and used for OC-SVM training and testing  on several
    values of 'nu' parameter using train_oc_svm(). The results are then printed.
    '''
    # Load SE and BSE ok and faulty data
    bse_ok_data = np.load("Data/BSE_ok.npy")
    bse_ok_data_extra = np.load("Data/BSE_ok_extra.npy")
    bse_faulty_data = np.load("Data/BSE_faulty_extended.npy")
    bse_ok_data_ld = np.load("Data/low_dim_BSE_ok.npy")
    bse_ok_data_extra_ld = np.load("Data/low_dim_BSE_ok_extra.npy")
    bse_faulty_data_ld = np.load("Data/low_dim_BSE_faulty_extended.npy")

    se_ok_data = np.load("Data/SE_ok.npy")
    se_ok_data_extra = np.load("Data/SE_ok_extra.npy")
    se_faulty_data = np.load("Data/SE_faulty_extended.npy")
    se_ok_data_ld = np.load("Data/low_dim_SE_ok.npy")
    se_ok_data_extra_ld = np.load("Data/low_dim_SE_ok_extra.npy")
    se_faulty_data_ld = np.load("Data/low_dim_SE_faulty_extended.npy")

    # Load the Encoding models
    AE_1 = load_model("Model_Saves/Detailed/OcSvm/encoder_extended_BasicAutoencoderEvenDeeperExtraLLR_e600_b4_detailed", compile=False)
    AE_2 = load_model("Model_Saves/Detailed/OcSvm/encoder_extended_BasicAutoencoderDeeper_e400_b4_detailed", compile=False)
    AE_3 = load_model("Model_Saves/Detailed/OcSvm/encoder_extended_BasicAutoencoderHFDeeperLLR_e400_b4_detailed", compile=False)
    AE_4 = load_model("Model_Saves/Detailed/OcSvm/encoder_extended_BasicAutoencoderLFDeeperLLR_e400_b4_detailed", compile=False)
    AE_5 = load_model("Model_Saves/Detailed/OcSvm/encoder_extended_HighStrideAutoencoderDeeper_e400_b4_detailed", compile=False)
    AE_6 = load_model("Model_Saves/Detailed/OcSvm/encoder_extended_HighStrideTransposeConvAutoencoder_e400_b4_detailed", compile=False)
    AE_7 = load_model("Model_Saves/Detailed/OcSvm/encoder_extended_TransposeConvAutoencoder_e400_b4_detailed", compile=False)

    # Lists to hold arrays and their shortened names
    models = [AE_1, AE_2, AE_3, AE_4, AE_5, AE_6, AE_7]
    model_names = ["Basic_Even_Deeper", "Basic", "HF_Deeper", "LF_Deeper",
                   "High_Stride", "High_Stride_Transpose", "Transpose"]

    # Values of 'nu' parameter to test and find the best from
    nu_values = [0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001]

    # Extract BSE and SE features for each model and test through several 'nu' values
    for model, name in zip(models, model_names):
        print("\n#MODEL")
        print(name + "\n")
        # Extract BSE and then SE features
        bse_ok_data_features = extract_features(bse_ok_data, model)
        bse_faulty_data_features = extract_features(bse_faulty_data, model)
        bse_ok_data_extra_features = extract_features(bse_ok_data_extra, model)

        se_ok_data_features = extract_features(se_ok_data, model)
        se_faulty_data_features = extract_features(se_faulty_data, model)
        se_ok_data_extra_features = extract_features(se_ok_data_extra, model)
        # For each 'nu' parameter value, train and test the oc-svm model
        for nu_value in nu_values:
            print("\nValue of nu: " + str(nu_value) + "\n")
            print("BSE:")
            train_oc_svm(bse_ok_data_features, bse_faulty_data_features,
                         bse_ok_data_extra_features, nu_val=nu_value)
            print("\nSE:")
            train_oc_svm(se_ok_data_features, se_faulty_data_features,
                         se_ok_data_extra_features, nu_val=nu_value)


    # Load the BSE embedding models
    siamese_BSE = load_model("Model_Saves/Detailed/OcSvm/embedding_BasicSiameseNetLowerDropout_BSE_extended_e60_b4_detailed", compile=False)
    print("\n#MODEL")
    print("Basic Siamese lowDrop BSE:")
    # Evaluate the model on several nu values
    model_eval(bse_ok_data, bse_ok_data_extra, bse_faulty_data, siamese_BSE, nu_values)

    siamese_BSE = load_model("Model_Saves/Detailed/OcSvm/embedding_SiameseNetLiteMultipleConvAltTwo_BSE_extended_e40_b4_detailed", compile=False)
    print("\n#MODEL")
    print("Siamese Lite MultipleConv AltTwo BSE:")
    model_eval(bse_ok_data, bse_ok_data_extra, bse_faulty_data, siamese_BSE, nu_values)

    # Load the SE embedding model
    siamese_SE = load_model("Model_Saves/Detailed/OcSvm/embedding_SiameseNetLiteMultipleConv_SE_extended_e40_b4_detailed", compile=False)
    print("\n#MODEL")
    print("Siamese Lite MultipleConv SE:")
    model_eval(se_ok_data, se_ok_data_extra, se_faulty_data, siamese_SE, nu_values)

    # Now for low dim siamese models
    globals_list = globals()
    globals_list['IMG_WIDTH'] = 384
    globals_list['IMG_HEIGHT'] = 384

    siamese_BSE = load_model("Model_Saves/Detailed/OcSvm/embedding_low_dims_SiameseNetLiteMultipleConvWithoutDropout_BSE_extended_e40_b4_detailed", compile=False)
    print("\n#MODEL")
    print("low dim Siamese Lite MultipleConv withoutDrop BSE:")
    model_eval(bse_ok_data_ld, bse_ok_data_extra_ld, bse_faulty_data_ld, siamese_BSE, nu_values)

    siamese_SE = load_model("Model_Saves/Detailed/OcSvm/embedding_low_dims_SiameseNetLiteMultipleConvWithoutDropout_SE_extended_e40_b4_detailed", compile=False)
    print("\n#MODEL")
    print("low dim Siamese Lite MultipleConv withoutDrop SE:")
    model_eval(se_ok_data_ld, se_ok_data_extra_ld, se_faulty_data_ld, siamese_SE, nu_values)

    # Now the triplet models
    triplet_BSE = load_model("Model_Saves/Detailed/OcSvm/embedding_low_dim_BasicTripletNetWithoutDropout_BSE_set_3_e60_b4_detailed", compile=False)
    print("\n#MODEL")
    print("low dim Basic Triplet Net withoutDrop BSE:")
    model_eval(bse_ok_data_ld, bse_ok_data_extra_ld, bse_faulty_data_ld, triplet_BSE, nu_values)

    triplet_BSE = load_model("Model_Saves/Detailed/OcSvm/embedding_low_dim_BasicTripletNetDeeperWithoutDropout_BSE_set_1_e60_b4_detailed", compile=False)
    print("\n#MODEL")
    print("low dim Basic Triplet Deeper withoutDrop BSE:")
    model_eval(bse_ok_data_ld, bse_ok_data_extra_ld, bse_faulty_data_ld, triplet_BSE, nu_values)

    triplet_BSE = load_model("Model_Saves/Detailed/OcSvm/embedding_low_dim_TripletNetMultipleConvWithoutDropout_BSE_set_1_e60_b4_detailed", compile=False)
    print("\n#MODEL")
    print("low dim Triplet Net Multiple Conv withoutDrop BSE:")
    model_eval(bse_ok_data_ld, bse_ok_data_extra_ld, bse_faulty_data_ld, triplet_BSE, nu_values)

    triplet_BSE = load_model("Model_Saves/Detailed/OcSvm/embedding_low_dim_BasicTripletNetHFWithoutDropout_BSE_set_1_e60_b4_detailed", compile=False)
    print("\n#MODEL")
    print("low dim Basic Triplet Net HF withoutDrop BSE:")
    model_eval(bse_ok_data_ld, bse_ok_data_extra_ld, bse_faulty_data_ld, triplet_BSE, nu_values)

    triplet_SE = load_model("Model_Saves/Detailed/OcSvm/embedding_low_dim_BasicTripletNetWithoutDropout_SE_set_2_e60_b4_detailed", compile=False)
    print("\n#MODEL")
    print("low dim Basic Triplet Net withoutDrop SE:")
    model_eval(se_ok_data_ld, se_ok_data_extra_ld, se_faulty_data_ld, triplet_SE, nu_values)

    triplet_SE = load_model("Model_Saves/Detailed/OcSvm/embedding_low_dim_BasicTripletNetDeeperWithoutDropout_SE_set_1_e60_b4_detailed", compile=False)
    print("\n#MODEL")
    print("low dim Basic Triplet Deeper withoutDrop SE:")
    model_eval(se_ok_data_ld, se_ok_data_extra_ld, se_faulty_data_ld, triplet_SE, nu_values)

    triplet_SE = load_model("Model_Saves/Detailed/OcSvm/embedding_low_dim_TripletNetMultipleConvWithoutDropout_SE_set_1_e60_b4_detailed", compile=False)
    print("\n#MODEL")
    print("low dim Triplet Net Multiple Conv withoutDrop SE:")
    model_eval(se_ok_data_ld, se_ok_data_extra_ld, se_faulty_data_ld, triplet_SE, nu_values)

    triplet_SE = load_model("Model_Saves/Detailed/OcSvm/embedding_low_dim_BasicTripletNetHFWithoutDropout_SE_set_3_e60_b4_detailed", compile=False)
    print("\n#MODEL")
    print("low dim Basic Triplet Net HF withoutDrop SE:")
    model_eval(se_ok_data_ld, se_ok_data_extra_ld, se_faulty_data_ld, triplet_SE, nu_values)



def train_oc_svm(ok_data_features, faulty_data_features,
                 testing_ok_data_features, nu_val=0.02):
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
        nu_val: OC-SVM parameter, refer to:
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    '''
    # Create the OC-SVM model, gamma='auto' vs 'scale'
    oc_svm_model = OneClassSVM(gamma='auto', nu=nu_val)
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


def model_eval(ok_data, ok_data_extra, faulty_data, model, nu_values):
    '''
    Helper method for evaluating siamese nets. Extracts features from given images
    using given model and then tests the performance of OC-SVM on extracted features
    through all nu values given.

    Arguments:
        ok_data: A numpy array of float32 [0,1] values representing training OK images.
        ok_data_extra: A numpy array of float32 [0,1] values representing testing
        OK images.
        faulty_data: A numpy array of float32 [0,1] values representing testing
        Faulty images.
        model: An instantiated pretrained embedding or encoding model.
        nu_values: A list of nu values for OC-SVM classifier.
    '''
    # Extract features using given model
    ok_data_features = extract_features(ok_data, model)
    faulty_data_features = extract_features(faulty_data, model)
    ok_data_extra_features = extract_features(ok_data_extra, model)
    # Train and test OC-SVM through several nu values
    for nu_value in nu_values:
        print("\nValue of nu: " + str(nu_value) + "\n")
        train_oc_svm(ok_data_features, faulty_data_features,
                     ok_data_extra_features, nu_val=nu_value)


if __name__ == "__main__":
    main()
