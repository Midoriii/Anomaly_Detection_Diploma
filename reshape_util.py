import numpy as np

from cv2 import cv2


def crop_reshape(images):
    images_list = []

    for image in images:
        img = cv2.imread(image)
        # Resize into 768*768 if bigger
        resized = cv2.resize(img, (768, 840))
        # Crop the bottom info
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
    # Normalize the data
    normalized = images.astype('float32') / 255.0
    # Reshape into desirable shape
    reshaped = normalized.reshape(normalized.shape[0], img_width, img_height, 1)

    return reshaped
