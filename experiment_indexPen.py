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

isDataGen = True

for i, path in enumerate(specimen_list):
    # generate orignial data
    print('Processing specimen #' + str(i) + '__________________________________')
    radar_data_grapher_volumned(path, isCluster=True, augmentation=['clipping'], isDataGen=isDataGen)
    radar_data_grapher_volumned(path, isCluster=True, augmentation=['trans', 'clipping'], isDataGen=isDataGen)
    radar_data_grapher_volumned(path, isCluster=True, augmentation=['rot', 'clipping'], isDataGen=isDataGen)
    radar_data_grapher_volumned(path, isCluster=True, augmentation=['scale', 'clipping'], isDataGen=isDataGen)

    radar_data_grapher_volumned(path, isCluster=True, augmentation=['trans', 'rot', 'clipping'], isDataGen=isDataGen)
    radar_data_grapher_volumned(path, isCluster=True, augmentation=['trans', 'scale', 'clipping'], isDataGen=isDataGen)
    radar_data_grapher_volumned(path, isCluster=True, augmentation=['rot', 'scale', 'clipping'], isDataGen=isDataGen)

    radar_data_grapher_volumned(path, isCluster=True, augmentation=['trans', 'rot', 'scale', 'clipping'], isDataGen=isDataGen)



