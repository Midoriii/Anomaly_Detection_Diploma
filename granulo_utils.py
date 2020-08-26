import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2
from PIL import Image
from scipy import ndimage



def reshape_img_from_float(image, width, height):
    # Stretch out the values
    reshaped_image = image * 255.0
    # Reshape to desired size
    reshaped_image = reshaped_image.reshape(width, height)
    # Convert array to Image .. a roundabout way of transforming float32->uint8
    reshaped_image = Image.fromarray(reshaped_image)
    # Convert to grayscale
    reshaped_image = reshaped_image.convert("L")
    # Convert back to array
    return np.array(reshaped_image)


def threshold_image(image, threshold):
    # Threshold the image .. INV since we want to count white pixels that have
    # actual value in granulometry
    ret, thresh = cv2.threshold(np.array(image), threshold, 255, cv2.THRESH_BINARY_INV)
    # Show the thresholded image
    #cv2.imshow("Thresholded pic", thresh)
    #cv2.imshow("Orig pic", image)
    #cv2.waitKey(0)
    return thresh


def plot_histogram(image):
    # Get Histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Plot it
    plt.plot(hist)
    plt.xlim([0, 256])
    # Show histogram
    plt.show()


def show_opening_contours(image, opening_element, label=""):
    # Do the opening itself
    opening = ndimage.binary_opening(image, structure=opening_element)
    # Plot the opening
    plt.figure(figsize=(12, 12))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title("Thresholded - " + label)
    plt.subplot(122)
    plt.imshow(opening, cmap='gray')
    plt.title("Opened")
    plt.contour(opening, [0.5], colors='b', linewidths=2)
    plt.show()


def granulometry_score(image, opening_element):
    # Do the opening itself
    opening = ndimage.binary_opening(image, structure=opening_element)
    # Calc how many pixels of the image remain
    granulo = opening.sum()
    return granulo
