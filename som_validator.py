import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style

from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN

from scipy.spatial import distance

from utils import generate_plot

# zl path
# radarData_path = '/Users/hanfei/data/072819_zl_onNotOn/f_data-2019-07-28_22-11-01.258054_zl_onNotOn_rnn/f_data.p'
# videoData_path = '/Users/hanfei/data/072819_zl_onNotOn/v_data-2019-07-28_22-10-32.249041_zl_onNotOn_rnn/cam1'
# mergedImg_path = '/Users/hanfei/figures/new'
# raw_path = 'F:/onNotOn_raw/zl_onNoton_raw.p'

# ag path
# radarData_path = '/Users/hanfei/data/072819_ag_onNotOn/f_data-2019-07-28_21-44-17.102820_ag_onNotOn_rnn/f_data.p'
# videoData_path = '/Users/hanfei/data/072819_ag_onNotOn/v_data-2019-07-28_21-44-08.514321_ag_onNotOn_rnn/cam1'
# mergedImg_path = '/Users/hanfei/figures/ag_onNotOn_x03y03z03_clustered_esp02ms4'
# raw_path = 'F:/onNotOn_raw/ag_onNoton_raw.p'

# zy path
radarData_path = '/Users/hanfei/data/072919_zy_onNotOn/f_data.p'
videoData_path = '/Users/hanfei/data/072919_zy_onNotOn/v_data-2019-07-29_11-40-34.810544_zy_onNotOn/cam1'
mergedImg_path = '/Users/hanfei/figures/zy_onNotOn_x03y03z03_clustered_esp02ms4'
raw_path = 'F:/onNotOn_raw/zy_onNoton_raw.p'

# utility directory to save the pyplots
radar_3dscatter_path = '/Users/hanfei/figures/plots'

radar_data = list(pickle.load(open(radarData_path, 'rb')).items())
radar_data.sort(key=lambda x: x[0])  # sort by timestamp
videoData_list = os.listdir(videoData_path)
videoData_timestamps = list(map(lambda x: float(x.strip('.jpg')), videoData_list))

style.use('fivethirtyeight')
white_color = 'rgb(255, 255, 255)'
black_color = 'rgb(0, 0, 0)'
red_color = 'rgb(255, 0, 0)'

DBSCAN_esp = 0.2
DBSCAN_minSamples = 4

# input data for the classifier that has the shape n*4*100, n being the number of samples
num_padding = 50
data_for_classifier = np.zeros((len(radar_data), num_padding, 4))
data_for_classifier_flattened = np.zeros((len(radar_data), 1, 4 * num_padding + 1))

fnt = ImageFont.truetype("Arial.ttf", 16)

generate_plot(radar_data, videoData_timestamps, videoData_path, DBSCAN_esp, DBSCAN_minSamples, num_padding, fnt, radar_3dscatter_path, mergedImg_path)
