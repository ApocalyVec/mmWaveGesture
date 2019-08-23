import os
import pickle

import numpy as np
from utils.path_utils import radar_data_grapher_volumned, generate_path

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
    radar_data_grapher_volumned(path, isCluster=True, isDataGen=isDataGen)
    radar_data_grapher_volumned(path, isCluster=True, augmentation=['trans'], isDataGen=isDataGen)
    radar_data_grapher_volumned(path, isCluster=True, augmentation=['rot'], isDataGen=isDataGen)
    radar_data_grapher_volumned(path, isCluster=True, augmentation=['scale'], isDataGen=isDataGen)

    radar_data_grapher_volumned(path, isCluster=True, augmentation=['trans', 'rot'], isDataGen=isDataGen)
    radar_data_grapher_volumned(path, isCluster=True, augmentation=['trans', 'scale'], isDataGen=isDataGen)
    radar_data_grapher_volumned(path, isCluster=True, augmentation=['rot', 'scale'], isDataGen=isDataGen)

    radar_data_grapher_volumned(path, isCluster=True, augmentation=['trans', 'rot', 'scale'], isDataGen=isDataGen)

# add zero valued class
num_class_samples = 800
data_shape = (100, 1, 25, 25, 25)
label_dict = pickle.load(open('D:/indexPen/labels/label_dict.p', 'rb'))
dummy_class_label = 5
# generate the data
dataset_path = 'D:/indexPen/dummyset'
for i in range(num_class_samples):
    print('Processing ' + str(i) + ' of ' + str(num_class_samples))
    dummy_name = str(i) + '_dummy'
    dummy_fn = os.path.join(dataset_path, dummy_name + '.npy')
    np.save(dummy_fn, np.zeros(data_shape))
    label_dict[dummy_name] = dummy_class_label