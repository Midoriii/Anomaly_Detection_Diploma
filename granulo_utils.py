'''
A util module containing useful functions for performing granulometry related
tasks on the given data.


func: reshape_img_from_float(image, width, height): Denormalizes image from float32
[0,1] to uint8 [0,255] and reshapes to dimensions (width, height).

func: threshold_image(image, threshold): Performs binary thresholding on image
with given threshold.

func: remove_center(image): Removes 50x50 central area from image.

func: plot_histogram(image): Calculates and shows the histogram of image.

func: show_opening_contours(image, element, label): Shows thresholded image vs
opened thresholded one with contours around the opening.

func: granulometry_score(image, element): Performs granulometry on image,
returns the sum of remaining white pixels after binary opening.
'''

import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2
from PIL import Image
from scipy import ndimage



def reshape_img_from_float(image, width, height):
    '''
    A helper method to denormalize and reshape previously saved image, turning it
    from float32 in [0,1] to grayscale uint8 in [0,255] and reshaping it to
    given dimensions for further processing.

    Arguments:
        image: A numpy array representing a 2D image, should be normalized.
        width: Integer, desired width dimension.
        height: Integer, desired height dimension.

    Returns:
        reshaped_image: A numpy array of uint8 values in range [0,255] shaped
        into (width, height).
    '''
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
    '''
    Method that thresholds given image with binary thresholding, where values
    below given threshold are set to 255 and values above are set to 0. This is
    because granulometry works by counting remaining pixels, and in our data
    the stains that we want to isolate and count are black or dark grey, hence
    we need to set them to white (255), so granulometry can sum those pixels up,
    after the binary opening is performed.

    Arguments:
        image: A numpy array representing a 2D image, in grayscale.
        threshold: Integer value for binary thresholding.

    Returns:
        thresh: A numpy array containing the thresholded binary image.
    '''
    # Threshold the image .. INV since we want to count white pixels that have
    # actual value in granulometry
    ret, thresh = cv2.threshold(np.array(image), threshold, 255, cv2.THRESH_BINARY_INV)
    # Show the thresholded image
    #cv2.imshow("Thresholded pic", thresh)
    #cv2.imshow("Orig pic", image)
    #cv2.waitKey(0)
    return thresh


def remove_center(image):
    '''
    Helper function to remove the center 50 by 50 part of images. This is because
    this part contains the central hole that degraded granulometry performance.
    The method expects images of size 768x768, which might be changed in the future.

    Arguments:
        image: A numpy array representing a 2D image with dimensions of 768x768.

    Returns:
        image: A numpy array identical to the given image with central 50x50 area
        filled with 0s, effectively removing the area from granulometry, which
        counts pixels with value > 0.
    '''
    image[354:404, 354:404] = 0
    return image


def plot_histogram(image):
    '''
    Just a simple histogram visualisation of a given image. Works in grayscale.

    Arguments:
        image: A numpy array representing a 2D image.
    '''
    # Get Histogram of channel 0, since we're working with grayscale
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Plot it
    plt.plot(hist)
    plt.xlim([0, 256])
    # Show histogram
    plt.show()


def show_opening(image, opening_element, label=""):
    '''
    A visualising method that shows the original thresholded image and the result
    of binary opening with given element on the picture.

    Arguments:
        image: A preferably thresholded image on which the opening will be performed,
        given as a numpy array.
        opening_element: Array of 0s and 1s representing the element with which
        to perform the opening.
        label: Additional string, helps with distinguishing what kind of picture
        is being worked with. I'm using it to distinguish between OK and faulty pics.
    '''
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
    # Plot the contours around the opening
    #plt.contour(opening, [0.5], colors='b', linewidths=2)
    plt.show()


def granulometry_score(image, opening_element):
    '''
    Preforms binary opening on given image and afterwards sums up the pixels.
    Binary opening returns only 0s and 1s (False and True), so by simply
    summing the pixels one can get a sort of 'score' of how much of the picture
    remained after performing opening.

    Arguments:
        image: A numpy array representing a 2D image.
        opening_element: Array of 0s and 1s representing an element with which
        to perform the opening.

    Returns:
        granulo: An integer representing how many pixels remained after opening.
    '''
    # Do the opening itself
    opening = ndimage.binary_opening(image, structure=opening_element)
    # Calc how many pixels of the image remain
    granulo = opening.sum()
    return granulo
