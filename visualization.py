import pickle
import numpy as np

import os

f_data_fn = 'data/fdata/f_data-2019-06-29_18-30-36.891690/f_data.p'
v_data_cam1_fn = 'data/vdata/v_data-2019-06-29_18-30-01.876610/cam1'

f_data = pickle.load(open(f_data_fn, 'rb'))

f_timestamps = np.asarray(list(f_data.keys()))
v_timestamps = np.asarray(list(float(x.strip('.jpg')) for x in os.listdir(v_data_cam1_fn)))

v_filler = np.zeros_like(v_timestamps)
f_filler = list(map(lambda x: x+.1, np.zeros_like(f_timestamps)))

import matplotlib
import PyQt5

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

# w = 7195
# h = 3841

plt.ion()

plt.scatter(f_timestamps, f_filler, color='r', label='Radar Data Timestamps')
plt.scatter(v_timestamps, v_filler, color='b', label='Video Data Timestamps')

# my_dpi=150
# # plt.figure(figsize=(1500, 1500))
# # plt.savefig('diff.png')