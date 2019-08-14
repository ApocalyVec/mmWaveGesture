import pickle
import numpy as np

import os

# verify the timestamp overlap between vData and fData

f_data_fn = 'F:/indexPen/data/f_data_zy_1/f_data.p'
v_data_fn = 'F:/indexPen/data/v_data_zy_1/cam2'

f_data = pickle.load(open(f_data_fn, 'rb'))

f_timestamps = np.asarray(list(f_data.keys()))
v_timestamps = np.asarray(list(float(x.strip('.jpg')) for x in os.listdir(v_data_fn)))

v_filler = np.zeros_like(v_timestamps)
f_filler = list(map(lambda x: x+.1, np.zeros_like(f_timestamps)))

import matplotlib
import PyQt5

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

plt.ion()

plt.scatter(f_timestamps, f_filler, color='r', label='Radar Data Timestamps')
plt.scatter(v_timestamps, v_filler, color='b', label='Video Data Timestamps')

plt.draw()