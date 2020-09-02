'''
The purpose of this script it to take the 5 BSE and 5 SE hand-picked prototype
images and turn them into the same shape and format as the rest of the data.

Prototype images are resized to 768x768, the info bar is cropped off. Afterwards
the images are normalized to float32 in range [0,1] and reshaped into Keras Input
shape of (len(images), width, height, 1). Finally they are saved for further use
during anomaly detection with siamese networks.
'''
import glob
import numpy as np
import cv2

from reshape_util import crop_reshape
from reshape_util import reshape_normalize


IMG_WIDTH = 768
IMG_HEIGHT = 768

proto_images_se = glob.glob('Clonky-prototypy/*_3*')
proto_images_bse = glob.glob('Clonky-prototypy/*_4*')

proto_images_se_list = crop_reshape(proto_images_se)
proto_images_bse_list = crop_reshape(proto_images_bse)

proto_images_se_list = reshape_normalize(proto_images_se_list, IMG_WIDTH, IMG_HEIGHT)
proto_images_bse_list = reshape_normalize(proto_images_bse_list, IMG_WIDTH, IMG_HEIGHT)

print(proto_images_se_list.shape)
print(proto_images_bse_list.shape)

np.save("Data/SE_prototypes.npy", proto_images_se_list)
np.save("Data/BSE_prototypes.npy", proto_images_bse_list)
