import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

ag = np.load('D:/Programing/19Sum_ft_localization/darkflow-master/ft_detection/Predict_tracking/tracking_radar/radar_arrays/ag_radar_tracking.npy')
zl = np.load('D:/Programing/19Sum_ft_localization/darkflow-master/ft_detection/Predict_tracking/tracking_radar/radar_arrays/zl_radar_tracking.npy')
zy = np.load('D:/Programing/19Sum_ft_localization/darkflow-master/ft_detection/Predict_tracking/tracking_radar/radar_arrays/zy_radar_tracking.npy')

all = np.concatenate((ag, zl, zy))
mmScaler = MinMaxScaler()
all[:, 3:] = mmScaler.fit_transform(all[:, 3:])
pickle.dump(mmScaler, open('D:/Programing/19Sum_ft_localization/mmWaveGesture/ThuMouse/scaler/mmScaler', 'wb'))

label_dict = dict()

for row in all:
    if str(row[0].as_integer_ratio()[0]) + '_' + str(row[0].as_integer_ratio()[1]) in label_dict.keys():
        raise Exception('Duplicate key')

    label_dict[str(row[0].as_integer_ratio()[0]) + '_' + str(row[0].as_integer_ratio()[1])] = np.asarray([row[3], row[4]])
