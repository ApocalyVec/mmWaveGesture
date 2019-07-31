import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style

from PIL import Image, ImageDraw

radarData_path =  'data/072819_zl_onNotOn/f_data-2019-07-28_22-11-01.258054_zl_onNotOn_rnn/f_data.p'
videoData_path = 'data/072819_zl_onNotOn/v_data-2019-07-28_22-10-32.249041_zl_onNotOn_rnn/cam1'
mergedImg_path = 'test'
radarImg_path = 'radar_test'

radar_data = list(pickle.load(open(radarData_path, 'rb')).items())
radar_data.sort(key=lambda x: x[0])  # sort by timestamp
videoData_list = os.listdir(videoData_path)
videoData_timestamps = list(map(lambda x: float(x.strip('.jpg')), videoData_list))

style.use('fivethirtyeight')
color = 'rgb(255, 255, 255)'
for timestamp, data in radar_data:
    i = radar_data.index((timestamp, data))
    print('Processing ' + str(i + 1) + ' of ' + str(len(radar_data)))

    closest_video_timestamp = min(videoData_timestamps,
                                  key=lambda x: abs(x - timestamp))
    closest_video_path = os.path.join(videoData_path, str(closest_video_timestamp) + '.jpg')
    closest_video_img = Image.open(closest_video_path)

    # plot the radar scatter
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['x'], data['y'], data['z'], c=data['doppler'], marker='D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(os.path.join(radarImg_path, str(timestamp) + '.jpg'))
    plt.clf()

    radar_img = Image.open(os.path.join(radarImg_path, str(timestamp) + '.jpg'))

    images = [closest_video_img, radar_img]

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    timestamp_difference = abs(float(timestamp) - float(closest_video_timestamp))
    draw = ImageDraw.Draw(new_im)

    # draw the timestamp difference on the image
    (x, y) = (50, 70)
    message = "Timestamp Difference, abs(rt-vt): " + str(timestamp_difference)
    draw.text((x,y), message, fill=color)

    # draw the timestamp
    (x, y) = (50, 50)
    message = "Timestamp: " + str(timestamp)
    draw.text((x, y), message, fill=color)

    new_im.save(os.path.join(mergedImg_path, str(timestamp) + '.jpg'))

###################################

label_radar_data = []

(x, y) = (50, 30)

for i in range(len(radar_data)):
    timestamp = radar_data[i][0]
    data = radar_data[i][1]

    print('Radar Image ' + str(i + 1) + ' of ' + str(len(radar_data)) + '   : ' + str(timestamp), end='')

    im = Image.open(os.path.join(mergedImg_path, str(timestamp) + '.jpg'))
    draw = ImageDraw.Draw(im)
    message = 'Current Image'
    color = 'rgb(255, 255, 255)'
    draw.text((x, y), message, fill=color)
    im.show()

    im = Image.open(os.path.join(mergedImg_path, str(radar_data[i+1][0]) + '.jpg'))
    draw = ImageDraw.Draw(im)
    message = 'Previous Image'
    color = 'rgb(255, 0, 255)'
    draw.text((x, y), message, fill=color)
    im.show()

    im = Image.open(os.path.join(mergedImg_path, str(radar_data[i-1][0]) + '.jpg'))
    draw = ImageDraw.Draw(im)
    message = 'Next Image'
    color = 'rgb(0, 255, 255)'
    draw.text((x, y), message, fill=color)
    im.show()

    while 1:
        label = int(input(' Label 0 for unchanged, 1 for lifting, and 2 for setting'))
        if label == 1 or label == 2 or label == 0:
            break
        else:
            print('Label must be 0, 1 or 2; your input is ' + str(label))

    label_radar_data.append((timestamp, data, label))
