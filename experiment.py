# zr 0 ######################################################
import os

from utils import radar_data_grapher_volumned, generate_path
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D

specimen_list = {generate_path('zr', 0), generate_path('zr', 1),
                 generate_path('py', 0), generate_path('py', 1),
                 generate_path('ya', 0), generate_path('ya', 1),
                 generate_path('zl', 0), generate_path('zl', 1),
                 generate_path('zy', 0), generate_path('zy', 1)
                 }

# use data augmentation

for path in specimen_list:
    # generate orignial data
    radar_data_grapher_volumned(path, isCluster=True)
    radar_data_grapher_volumned(path, isCluster=True, augmentation='trans', out_name='trans_aug')
