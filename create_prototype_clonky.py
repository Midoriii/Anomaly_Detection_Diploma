import glob
import numpy as np
import cv2
from siamese_network_data_prep_by_method import crop_reshape


img_width = 768
img_height = 768

proto_images_se = glob.glob('Clonky-prototypy/*_3*')
proto_images_bse = glob.glob('Clonky-prototypy/*_4*')

proto_images_se_list = crop_reshape(proto_images_se)
proto_images_bse_list = crop_reshape(proto_images_bse)

proto_images_se_list = proto_images_se_list.astype('float32') / 255.0
proto_images_bse_list = proto_images_bse_list.astype('float32') / 255.0

proto_images_se_list = proto_images_se_list.reshape(proto_images_se_list.shape[0], img_width, img_height, 1)
proto_images_bse_list = proto_images_bse_list.reshape(proto_images_bse_list.shape[0], img_width, img_height, 1)

print(proto_images_se_list.shape)
print(proto_images_bse_list.shape)

np.save("Data/SE_prototypes.npy", proto_images_se_list)
np.save("Data/BSE_prototypes.npy", proto_images_bse_list)
