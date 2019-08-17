import os

from utils import radar_data_grapher_volumned, generate_path, radar_data_grapher_volumned_track
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D

specimen_list = {generate_path('ag', 0, mode='thumouse'),
                 generate_path('zy', 0, mode='thumouse'),
                 generate_path('zl', 0, mode='thumouse')
                 }

# use data augmentation

isDataGen = True

for i, path in enumerate(specimen_list):
    # generate orignial data
    print('Processing specimen #' + str(i) + '__________________________________')
    # radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['trans'])
    # radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['rot'])
    # radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['scale'])
    radar_data_grapher_volumned_track(path, isPlot=True, isCluster=True)




