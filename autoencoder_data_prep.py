import glob
import numpy as np

from reshape_util import crop_reshape
from reshape_util import reshape_normalize

IMG_WIDTH = 768
IMG_HEIGHT = 768

# Grab all the images
ok_images = glob.glob('Clonky-ok/*')
faulty_images = glob.glob('Clonky-vadne/*')
ok_filtered_images = glob.glob('Clonky-ok-filtered/*')
faulty_full_images = glob.glob('Clonky-vadne-full/*')

lists_of_images = [ok_images, ok_filtered_images, faulty_images, faulty_full_images]
file_names = ["OK.npy", "OK_filtered.npy", "Faulty.npy", "Faulty_extended.npy"]

# Go through the loaded images and crop them, reshape them into 768x768,
# Normalize them and finally save them
for image_list, file_name in zip(lists_of_images, file_names):
    cropped_data = crop_reshape(image_list)
    normalized_data = reshape_normalize(cropped_data, IMG_WIDTH, IMG_HEIGHT)
    np.save("Data/" + file_name, normalized_data)
