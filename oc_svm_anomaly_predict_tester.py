'''
bla
'''
import numpy as np
from cv2 import cv2

from PIL import Image
from sklearn.svm import OneClassSVM
from keras.models import load_model



def main():
    '''
    bla
    '''
    # Load BSE data (and low dim variant)
    bse_ok = np.load("Data/BSE_ok.npy")
    bse_ok_ld = np.load("Data/low_dim_BSE_ok.npy")
    bse_ok_extra = np.load("Data/BSE_ok_extra.npy")
    bse_ok_extra_ld = np.load("Data/low_dim_BSE_ok_extra.npy")
    bse_faulty = np.load("Data/BSE_faulty.npy")
    bse_faulty_ld = np.load("Data/low_dim_BSE_faulty.npy")
    # Load SE data (and low dim variant)
    se_ok = np.load("Data/SE_ok.npy")
    se_ok_ld = np.load("Data/low_dim_SE_ok.npy")
    se_ok_extra = np.load("Data/SE_ok_extra.npy")
    se_ok_extra_ld = np.load("Data/low_dim_SE_ok_extra.npy")
    se_faulty = np.load("Data/SE_faulty.npy")
    se_faulty_ld = np.load("Data/low_dim_SE_faulty.npy")

    # Load desired models

    # For each model and its best nu

        # Extract features using models

        # Create and train OC-SVM model

        # Plot problematic images

    # Same for low dim models



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


if __name__ == "__main__":
    main()
