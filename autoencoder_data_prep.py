'''
A script to create data from provided images for autoencoders. Doesn't
divide the images by type, BSE and SE are mixed together.

It takes in turn all of the OK images, hand-filtered best OK images,
faulty images without plugged central hole and completely all
of the faulty images, plugged center included.

Images are cropped uniformly to 768x768, the dimensions of most of the
provided images, and the info bar is removed. Afterwards they are normalized
to float32 in range [0,1] and reshaped into preferred Keras Input shape,
which is (len(images), width, height, 1).

The resulting data is stored for further use, currently mainly by autoencoders.
'''
import glob
import numpy as np

from reshape_util import crop_reshape
from reshape_util import reshape_normalize

IMG_WIDTH = 768
IMG_HEIGHT = 768

# Grab all the images
ok_images = glob.glob('Clonky-ok/*')
ok_images_extra = glob.glob('Clonky-ok-2/*')
faulty_images = glob.glob('Clonky-vadne/*')
ok_filtered_images = glob.glob('Clonky-ok-filtered/*')
faulty_full_images = glob.glob('Clonky-vadne-full/*')

lists_of_images = [ok_images, ok_images_extra, ok_filtered_images, faulty_images, faulty_full_images]
file_names = ["OK.npy", "OK_extra.npy", "OK_filtered.npy", "Faulty.npy", "Faulty_extended.npy"]

# Go through the loaded images and crop them, reshape them into 768x768,
# Normalize them and finally save them
for image_list, file_name in zip(lists_of_images, file_names):
    cropped_data = crop_reshape(image_list)
    normalized_data = reshape_normalize(cropped_data, IMG_WIDTH, IMG_HEIGHT)
    np.save("Data/" + file_name, normalized_data)
