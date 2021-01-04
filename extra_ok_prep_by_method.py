'''
Copyright (c) 2021, Štěpán Beneš


Short script to divide newly acquired OK data for testing purposes into
BSE and SE data as per usual.
'''
import glob
import numpy as np

from reshape_util import crop_reshape
from reshape_util import reshape_normalize



IMG_WIDTH = 768
IMG_HEIGHT = 768

# Glob the images from the folder
ok_images_se = glob.glob('Clonky-ok-2/*_3*')
ok_images_bse = glob.glob('Clonky-ok-2/*_4*')

# Remove the info bar from the images and reshape them into 768x768
ok_images_se_list = crop_reshape(ok_images_se)
ok_images_bse_list = crop_reshape(ok_images_bse)

# Save the normalized data
np.save("Data/SE_ok_extra.npy", reshape_normalize(ok_images_se_list, IMG_WIDTH, IMG_HEIGHT))
np.save("Data/BSE_ok_extra.npy", reshape_normalize(ok_images_bse_list, IMG_WIDTH, IMG_HEIGHT))
