'''
bla
'''
import numpy as np


for file in ["low_dim_SE_ok", "low_dim_SE_ok_extra", "low_dim_SE_faulty_extended",
             "low_dim_BSE_ok", "low_dim_BSE_ok_extra", "low_dim_BSE_faulty_extended"]:
    data = np.load("Data/" + file + ".npy")
    data = ((data * 255.0) - 127.5) / 127.5
    np.save("DataBigan/" + file + "_bigan.npy", data)
