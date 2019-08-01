import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style

from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN

from scipy.spatial import distance

radarData_path = 'data/072819_zl_onNotOn/f_data-2019-07-28_22-11-01.258054_zl_onNotOn_rnn/f_data.p'
videoData_path = 'data/072819_zl_onNotOn/v_data-2019-07-28_22-10-32.249041_zl_onNotOn_rnn/cam1'
mergedImg_path = 'E:/figures/zl_onNotOn_x03y03z03_clustered_esp02ms4_2'
radar_3dscatter_path = 'E:/figures/radar_3dscatter'
radar_3dscatter_clustered_path = 'E:/figures/radar_3dscatter_clustered'

radar_data = list(pickle.load(open(radarData_path, 'rb')).items())
radar_data.sort(key=lambda x: x[0])  # sort by timestamp
videoData_list = os.listdir(videoData_path)
videoData_timestamps = list(map(lambda x: float(x.strip('.jpg')), videoData_list))

style.use('fivethirtyeight')
white_color = 'rgb(255, 255, 255)'
black_color = 'rgb(0, 0, 0)'

DBSCAN_esp = 0.2
DBSCAN_minSamples = 4

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
    ax.set_xlim((-0.3, 0.3))
    ax.set_ylim((-0.3, 0.3))
    ax.set_zlim((-0.3, 0.3))
    ax.scatter(data['x'], data['y'], data['z'], c=data['doppler'], marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(os.path.join(radar_3dscatter_path, str(timestamp) + '.jpg'))
    plt.clf()

    # plot cluster ###############
    data = np.asarray([data['x'], data['y'], data['z']]).transpose()
    db = DBSCAN(eps=DBSCAN_esp, min_samples=DBSCAN_minSamples).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((-0.3, 0.3))
    ax.set_ylim((-0.3, 0.3))
    ax.set_zlim((-0.3, 0.3))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    clusters = []

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xyz = data[class_member_mask & core_samples_mask]
        # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', c=np.array([col]), s=28, marker='o')
        if xyz.any():  # in case there are none objects
            clusters.append(xyz)  # append this cluster data to the cluster list
        # each cluster is a 3 * n matrix
        xyz = data[class_member_mask & ~core_samples_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', c=np.array([col]), s=12, marker='X')

    #############################

    clusters_centers = list(map(lambda xyz: np.array([np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])]), clusters))
    clusters.sort(key=lambda xyz: distance.euclidean((0.0, 0.0, 0.0), np.array([np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])])))

    # find the center for each cluster
    # TODO making sure the closest cluster being colored consistent
    for xyz, col in zip(clusters, colors):
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', c=np.array([col]), s=28, marker='o')

    plt.savefig(os.path.join(radar_3dscatter_clustered_path, str(timestamp) + '.jpg'))

    #############################
    # Combine the three images
    radar_3dscatter_img = Image.open(os.path.join(radar_3dscatter_path, str(timestamp) + '.jpg'))
    radar_3dscatter_clustered_img = Image.open(os.path.join(radar_3dscatter_clustered_path, str(timestamp) + '.jpg'))

    images = [closest_video_img, radar_3dscatter_img, radar_3dscatter_clustered_img]  # add image here to arrange them horizontally
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
    draw.text((x,y), message, fill=white_color)
    # draw the timestamp
    (x, y) = (50, 50)
    message = "Timestamp: " + str(timestamp)
    draw.text((x, y), message, fill=white_color)

    # draw the number of points
    (x, y) = (640, 50)
    message = "Number of detected points: " + str(xyz.shape[0])
    draw.text((x, y), message, fill=black_color)

    # draw the number of clusters and number of noise point on the clutter plot
    (x, y) = (1280, 50)
    message = "Number of clusters: " + str(n_clusters_)
    draw.text((x, y), message, fill=black_color)
    (x, y) = (1280, 70)
    message = "Number of outliers: " + str(n_noise_)
    draw.text((x, y), message, fill=black_color)

    # save the combined image
    new_im.save(os.path.join(mergedImg_path, str(timestamp) + '.jpg'))
    plt.close('all')

# Following Code is for labeling ##################################

# label_radar_data = []
#
# (x, y) = (50, 30)
#
# for i in range(len(radar_data)):
#     timestamp = radar_data[i][0]
#     data = radar_data[i][1]
#
#     print('Radar Image ' + str(i + 1) + ' of ' + str(len(radar_data)) + '   : ' + str(timestamp), end='')
#
#     im = Image.open(os.path.join(mergedImg_path, str(timestamp) + '.jpg'))
#     draw = ImageDraw.Draw(im)
#     message = 'Current Image'
#     white_color = 'rgb(255, 255, 255)'
#     draw.text((x, y), message, fill=white_color)
#     im.show()
#
#     im = Image.open(os.path.join(mergedImg_path, str(radar_data[i+1][0]) + '.jpg'))
#     draw = ImageDraw.Draw(im)
#     message = 'Previous Image'
#     white_color = 'rgb(255, 0, 255)'
#     draw.text((x, y), message, fill=white_color)
#     im.show()
#
#     im = Image.open(os.path.join(mergedImg_path, str(radar_data[i-1][0]) + '.jpg'))
#     draw = ImageDraw.Draw(im)
#     message = 'Next Image'
#     white_color = 'rgb(0, 255, 255)'
#     draw.text((x, y), message, fill=white_color)
#     im.show()
#
#     while 1:
#         label = int(input(' Label 0 for unchanged, 1 for lifting, and 2 for setting'))
#         if label == 1 or label == 2 or label == 0:
#             break
#         else:
#             print('Label must be 0, 1 or 2; your input is ' + str(label))
#
#     label_radar_data.append((timestamp, data, label))
