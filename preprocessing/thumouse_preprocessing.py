import pandas as pd
import numpy as np

ag = np.load('/Users/Leo/Desktop/radar_arrays/ag_radar_tracking.npy')
zl = np.load('/Users/Leo/Desktop/radar_arrays/zl_radar_tracking.npy')
zy = np.load('/Users/Leo/Desktop/radar_arrays/zy_radar_tracking.npy')

label_dict = dict()

for ag_row, zl_row, zy_row in zip(ag, zl, zy):
    label_dict['ag_0_' + str(ag_row[0].as_integer_ratio()[0]) + '_' + str(ag_row[0].as_integer_ratio()[0])] = np.asarray[ag_row[3], ag_row[4]]
    label_dict['zl_0_' + str(zl_row[0].as_integer_ratio()[0]) + '_' + str(zl_row[0].as_integer_ratio()[0])] = np.asarray[zl_row[3], zl_row[4]]
    label_dict['zy_0_' + str(zy_row[0].as_integer_ratio()[0]) + '_' + str(zy_row[0].as_integer_ratio()[0])] = np.asarray[zy_row[3], zy_row[4]]

