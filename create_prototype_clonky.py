import glob
import numpy as np
import cv2

from reshape_util import crop_reshape
from reshape_util import reshape_normalize


img_width = 768
img_height = 768

proto_images_se = glob.glob('Clonky-prototypy/*_3*')
proto_images_bse = glob.glob('Clonky-prototypy/*_4*')

proto_images_se_list = crop_reshape(proto_images_se)
proto_images_bse_list = crop_reshape(proto_images_bse)

proto_images_se_list = reshape_normalize(proto_images_se_list, img_width, img_height)
proto_images_bse_list = reshape_normalize(proto_images_bse_list, img_width, img_height)

print(proto_images_se_list.shape)
print(proto_images_bse_list.shape)

np.save("Data/SE_prototypes.npy", proto_images_se_list)
np.save("Data/BSE_prototypes.npy", proto_images_bse_list)
