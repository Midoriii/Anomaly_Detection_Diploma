'''
A util module containing often used methods while saving / using data.


func: crop_reshape(images): As name suggests, crops and reshapes a list of images.

func: reshape_normalize(images, width, height): Normalizes input data and reshapes
into (len(images), width, height, 1) format suitable for NN input.
'''

import numpy as np

from cv2 import cv2


def crop_reshape(images):
    '''
    Crops the additional info bar from images, reshapes them into desired
    (768, 768) shape and converts to grayscale.

    Arguments:
        images: A list of images to perform operations on, should be 2D array.

    Returns:
        images_list: Numpy array of images converted to grayscale,
        shaped into (768,768), and with cropped off info bar.
    '''
    images_list = []

    for image in images:
        img = cv2.imread(image)
        # Resize into 768*840 if bigger
        resized = cv2.resize(img, (768, 840))
        # Crop the bottom info, making it 768x768
        cropped_img = resized[0:768, 0:768]
        # Make it actual grayscale
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        # Add the new grayscale image to the list
        images_list.append(gray)
    # As numpy array
    images_list = np.array(images_list)
    # Return the list for further operations
    return images_list


def reshape_normalize(images, img_width, img_height):
    '''
    Converts input image array into float32, normalizes it by dividing by 255.0,
    and reshapes into a Keras Input type of shape; (len(images), width, height, 1).

    Arguments:
        images: Numpy array of images to peform operations on.
        img_width: Integer, desired width dimension of images.
        img_height: Integer, desired height dimension of images.

    Returns:
        reshaped: A numpy array of float32, normalized and reshaped images.
    '''
    # Normalize the data
    normalized = images.astype('float32') / 255.0
    # Reshape into desirable shape
    reshaped = normalized.reshape(normalized.shape[0], img_width, img_height, 1)

    return reshaped
