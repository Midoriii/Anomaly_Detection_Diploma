import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys


img_width = 768
img_height = 768
threshold = 50

# Load OK BSE data
ok_data = np.load("Data/BSE_ok.npy")
# Load also the anomalous BSE data
faulty_data = np.load("Data/BSE_faulty.npy")

# For each image, reshape back into 768x768 and stretch, since it's normalized
for i in range (0, faulty_data.shape[0]):
    faulty_img = faulty_data[i] * 255.0
    faulty_img = faulty_img.reshape(img_width, img_height)
    # Convert to Image
    actual_img = Image.fromarray(faulty_img)
    # Convert to grayscale
    actual_img = actual_img.convert("L")
    # Threshold the image
    ret, thresh = cv2.threshold(np.array(actual_img), threshold, 255, cv2.THRESH_BINARY)
    # Get Histogram
    hist = cv2.calcHist([np.array(actual_img)], [0], None, [256], [0, 256])
    # Plot it
    plt.plot(hist)
    plt.xlim([0, 256])
    # Show the thresholded image
    cv2.imshow("Thresholded pic", thresh)
    cv2.imshow("Orig pic", np.array(actual_img))
    cv2.waitKey(0)
    # Show histogram
    plt.show()
