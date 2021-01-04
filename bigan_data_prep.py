'''
Copyright (c) 2021, Štěpán Beneš


This script loads already prepared low dimensional data and changes their normalization
from [0,1] to [-1,1], which is more suited for biGAN training.

This is the only change made, and the scripts relies on other data prep scripts
to provide the data, as it does not create it from the raw images, unlike the other
data prep scripts.
'''
import numpy as np
from PIL import Image


for file in ["low_dim_SE_ok", "low_dim_SE_ok_extra", "low_dim_SE_faulty_extended",
             "low_dim_BSE_ok", "low_dim_BSE_ok_extra", "low_dim_BSE_faulty_extended"]:
    data = np.load("Data/" + file + ".npy")
    data = ((data * 255.0) - 127.5) / 127.5
    np.save("DataBigan/" + file + "_bigan.npy", data)

    img_list = []

    for i in range(data.shape[0]):
        img = data[i].reshape(384, 384)
        img = Image.fromarray(img, mode='L')
        img = img.resize((384, 384))
        img_list.append(np.asarray(img))

    imgs = np.array(img_list)
    imgs = imgs.reshape(data.shape[0], 384, 384, 1)
    imgs = imgs.astype('float32')
    print(data.shape)
    print(imgs.shape)

    np.save("DataBigan/extra_" + file + "_bigan.npy", imgs)
