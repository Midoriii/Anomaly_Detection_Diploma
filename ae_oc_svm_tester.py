'''

'''
import numpy as np

from sklearn.svm import OneClassSVM


# Constants
IMG_WIDTH = 768
IMG_HEIGHT = 768

# This should be main
# Load SE or BSE ok and faulty data
bse_ok_data = np.load("Data/BSE_ok.npy")
bse_faulty_data = np.load("Data/BSE_faulty_extended.npy")

se_ok_data = np.load("Data/SE_ok.npy")
se_faulty_data = np.load("Data/SE_faulty_extended.npy")

# Load the Encoding or Embedding model

# This will be a method
# Get encodings of all OK images

# Train OC-SVM on those

# Predict on faulty data and additional ok data



if __name__ == "__main__":
    main()
