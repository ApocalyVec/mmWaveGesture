import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

ag = np.load('F:/thumouse/labels/radar_arrays/ag_radar_tracking.npy')
zl = np.load('F:/thumouse/labels/radar_arrays/zl_radar_tracking.npy')
zy = np.load('F:/thumouse/labels/radar_arrays/zy_radar_tracking.npy')

all = np.concatenate((ag, zl, zy))
mmScaler = MinMaxScaler()
all[:, 3:] = mmScaler.fit_transform(all[:, 3:])
pickle.dump(mmScaler, open('D:/thumouse/scaler/thm_scaler.p', 'wb'))

label_dict = dict()

aug_strings = ['_trans', '_rot', '_scale', '_trans_rot', '_trans_scale', '_rot_scale', '_trans_rot_scale']

for i, row in enumerate(all):
    print('Processing ' + str(i+1) + ' of ' + str(len(all)))
    if str(row[0].as_integer_ratio()[0]) + '_' + str(row[0].as_integer_ratio()[1]) in label_dict.keys():
        raise Exception('Duplicate key')

    if aug_strings is not None:
        for aug_str in aug_strings:
            label_dict[str(row[0].as_integer_ratio()[0]) + '_' + str(row[0].as_integer_ratio()[1]) + aug_str] = np.asarray([row[3], row[4]])

label_dict_path = 'D:/thumouse/labels_timestep_1_noClp/label_dict.p'
