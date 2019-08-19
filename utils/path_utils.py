import random
from itertools import product

import numpy as np
import math

import pickle
import os
import shutil

import matplotlib.pyplot as plt
from matplotlib import style

from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN

from scipy.spatial import distance
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from transformation import translate, get_index, rotateZ, rotateY, rotateX, scale


augmentation_list = ['trans', 'rot', 'scale']

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def get_outliers(array, m=2.):
    """
    :param array: list of values
    :return the list of outliers

    """
    data = np.asarray(array)

    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m].tolist()


def distance_to_origin(x, y):
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))
    # rtn = 0.0
    # for value in point:
    #     rtn = rtn + math.pow(value, 2)
    # return math.sqrt(rtn)


def generate_single_plot(plot_save_path, mergedImg_path, font, closest_video_img, closest_video_timestamp, xyz,
                         n_clusters_, n_noise_, timestamp, cluster):
    white_color = 'rgb(255, 255, 255)'

    cluster_flattened = cluster.reshape((-1))
    cluster_flattened = np.insert(cluster_flattened, 0, timestamp)
    cluster_flattened = cluster_flattened.reshape(1, -1)

    ax = plt.subplot(221, projection='3d')
    ax.set_xlim((-1.0, 1.0))
    ax.set_ylim((-1.0, 1.0))
    ax.set_zlim((-1.0, 1.0))
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title('CLosest Cluster', fontsize=10)

    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], c=cluster[:, 3], marker='o')

    #############################
    # Combine the three images
    plt.savefig(os.path.join(plot_save_path, str(timestamp) + '.jpg'))
    radar_3dscatter_img = Image.open(os.path.join(plot_save_path, str(timestamp) + '.jpg'))

    images = [closest_video_img, radar_3dscatter_img]  # add image here to arrange them horizontally
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
    (x, y) = (20, 10)
    message = "Timestamp Difference, abs(rt-vt): " + str(timestamp_difference)
    draw.text((x, y), message, fill=white_color, font=font)
    # draw the timestamp
    (x, y) = (20, 30)
    message = "Timestamp: " + str(timestamp)
    draw.text((x, y), message, fill=white_color, font=font)

    # draw the number of points
    (x, y) = (20, 60)
    message = "Number of detected points: " + str(xyz.shape[0])
    draw.text((x, y), message, fill=white_color, font=font)

    # draw the number of clusters and number of noise point on the clutter plot
    (x, y) = (20, 80)
    message = "Number of clusters: " + str(n_clusters_)
    draw.text((x, y), message, fill=white_color, font=font)
    (x, y) = (20, 100)
    message = "Number of outliers: " + str(n_noise_)
    draw.text((x, y), message, fill=white_color, font=font)

    # save the combined image
    new_im.save(os.path.join(mergedImg_path, str(timestamp) + '.jpg'))
    plt.close('all')

    return cluster_flattened


# this function returns the flattened version of radar_data with timestamp information
def generate_plot(radar_data, videoData_timestamps, videoData_path, DBSCAN_esp, DBSCAN_minSamples, num_padding, font,
                  radar_3dscatter_path, mergedImg_path):
    data_for_classifier_flattened = np.zeros((len(radar_data), 1, 4 * num_padding + 1))
    for i, radarFrame in enumerate(radar_data):

        timestamp, fData = radarFrame
        print('Processing ' + str(i + 1) + ' of ' + str(len(radar_data)))

        closest_video_timestamp = min(videoData_timestamps,
                                      key=lambda x: abs(x - timestamp))
        closest_video_path = os.path.join(videoData_path, str(closest_video_timestamp) + '.jpg')
        closest_video_img = Image.open(closest_video_path)

        # Do DBSCAN cluster ###############
        # Do cluster ###############
        # map the points to their doppler value, this is for retrieving the doppler value after clustering
        data = np.asarray([fData['x'], fData['y'], fData['z'], fData['doppler']]).transpose()
        doppler_dict = {}
        for point in data:
            doppler_dict[tuple(point[:3])] = point[3:]
        # get rid of the doppler for clustering TODO should we consider the doppler in clustering?
        data = data[:, :3]

        db = DBSCAN(eps=DBSCAN_esp, min_samples=DBSCAN_minSamples).fit(data)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

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
            # ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', c=np.array([col]), s=12, marker='X')  # plot the noise

        # find the center for each cluster
        clusters_centers = list(
            map(lambda xyz: np.array([np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])]), clusters))
        clusters.sort(key=lambda xyz: distance.euclidean((0.0, 0.0, 0.0), np.array(
            [np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])])))

        #############################
        # center normalize hand cluster
        # clear the hand cluster
        hand_cluster = []

        if len(clusters) > 0:
            hand_cluster = clusters[0]

            xmean = np.mean(hand_cluster[:, 0])
            xmin = np.min(hand_cluster[:, 0])
            xmax = np.max(hand_cluster[:, 0])

            ymean = np.mean(hand_cluster[:, 1])
            ymin = np.min(hand_cluster[:, 1])
            ymax = np.max(hand_cluster[:, 1])

            zmean = np.mean(hand_cluster[:, 2])
            zmin = np.min(hand_cluster[:, 2])
            zmax = np.max(hand_cluster[:, 2])

            # append back the doppler
            # doppler array for this frame
            point_num = hand_cluster.shape[0]

            doppler_array = np.zeros((point_num, 1))
            for j in range(point_num):
                doppler_array[j:, ] = doppler_dict[tuple(hand_cluster[j, :3])]
            hand_cluster = np.append(hand_cluster, doppler_array,
                                     1)  # TODO this part needs validation, are the put-back dopplers correct?

            # Do the Mean Normalization
            # avoid division by zero, check if all the elements in a column are the same
            if np.all(hand_cluster[:, 0][0] == hand_cluster[:, 0]) or xmin == xmax:
                hand_cluster[:, 0] = np.zeros((point_num))
            else:
                hand_cluster[:, 0] = np.asarray(list(map(lambda x: (x - xmean) / (xmax - xmin), hand_cluster[:, 0])))

            if np.all(hand_cluster[:, 1][0] == hand_cluster[:, 1]) or ymin == ymax:
                hand_cluster[:, 1] = np.zeros((point_num))
            else:
                hand_cluster[:, 1] = np.asarray(list(map(lambda y: (y - ymean) / (ymax - ymin), hand_cluster[:, 1])))

            if np.all(hand_cluster[:, 2][0] == hand_cluster[:, 2]) or zmin == zmax:
                hand_cluster[:, 2] = np.zeros((point_num))
            else:
                hand_cluster[:, 2] = np.asarray(list(map(lambda z: (z - zmean) / (zmax - zmin), hand_cluster[:, 2])))
            # pad to 50
            hand_cluster_padded = np.pad(hand_cluster, ((0, num_padding - point_num), (0, 0)), 'constant',
                                         constant_values=0)
        else:
            hand_cluster_padded = np.zeros((num_padding, 4))

        data_for_classifier_flattened[i] = generate_single_plot(radar_3dscatter_path, mergedImg_path, font,
                                                                closest_video_img, closest_video_timestamp, xyz,
                                                                n_clusters_, n_noise_, timestamp, hand_cluster_padded)
    return data_for_classifier_flattened


def label(folder_path, data_file):
    img_folder = os.listdir(folder_path)
    gesture_timestamp = dict()
    for gesture in img_folder:
        if not gesture == '.DS_Store':
            gesture_timestamp[gesture[0:1]] = os.listdir(os.path.join(folder_path, gesture))
            gesture_timestamp[gesture[0:1]].remove('.DS_Store')
            gesture_timestamp[gesture[0:1]] = list(
                map(lambda x: float(x.strip('.jpg')), gesture_timestamp[gesture[0:1]]))

    data = pd.read_csv(data_file)

    timestamp_set = set()
    for timestamp in range(len(data)):
        timestamp_set.add(data.loc[timestamp].iat[1])

    not_found = []
    for gesture, timestamps in gesture_timestamp.items():
        for timestamp in timestamps:
            if timestamp not in timestamp_set:
                not_found.append(timestamp)
                print(str(timestamp) + ' , which is in folder ' + str(gesture) + ' not found in csv file')
                # raise ValueError('found a timestamp that is not in the given csv file!')

    for timestamp in range(len(data)):
        print('labeling ' + str(timestamp) + ' of ' + str(len(data)))
        for gesture in gesture_timestamp:
            if data.loc[timestamp].iat[1] in gesture_timestamp[gesture]:
                data.loc[timestamp].iat[0] = float(gesture)
                break
    return (data, not_found)


# variables used in snapPointsToVolume
xmin, xmax = -0.5, 0.5
ymin, ymax = 0.0, 0.5
zmin, zmax = -0.5, 0.5

heatMin, heatMax = -1.0, 1.0
xyzScaler = MinMaxScaler().fit(np.array([[xmin, ymin, zmin],
                                         [xmax, ymax, zmax]]))
heatScaler = MinMaxScaler().fit(np.array([[heatMin],
                                          [heatMax]]))

# volumn.shape = (5, 5, 5)
def snapPointsToVolume(points, volume_shape, isClipping=False, radius=3, decay=0.8):
    """
    make sure volume is a square
    :param points: n * 4 array
    :param heat: scale 0 to 1
    :param volume:
    """
    assert len(volume_shape) == 3 and volume_shape[0] == volume_shape[1] == volume_shape[2]
    volume = np.zeros(volume_shape)

    if len(points) != 0:

        # filter out points that are outside the bounding box
        # using ABSOLUTE normalization


        points_filtered = []
        for p in points:
            if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax and zmin <= p[2] <= zmax:
                points_filtered.append(p)
        if len(points_filtered) == 0:
            return volume
        points_filtered = np.asarray(points_filtered)

        points_filtered[:, :3] = xyzScaler.transform(points_filtered[:, :3])
        points_filtered[:, 3:] = heatScaler.transform(points_filtered[:, 3:])

        size = volume_shape[0]  # the length of thesquare side
        axis = np.array((size - 1) * points_filtered[:, :3], dtype=int)  # size minus 1 for index starts at 0

        for i, row in enumerate(points_filtered):
            heat = row[3]

            volume[axis[i][0], axis[i][1], axis[i][2]] = volume[axis[i][0], axis[i][1], axis[i][2]] + heat

            if isClipping:
                point_to_clip = get_index(shape=volume_shape, index=(axis[i][0], axis[i][1], axis[i][2]), r=radius)
                for dist, ptc in point_to_clip:
                    if dist != 0.0:
                        factor = (radius - dist + 1) * decay /radius
                        volume[ptc[0], ptc[1], ptc[2]] = volume[ptc[0], ptc[1], ptc[2]] + heat * factor
    return volume


def radar_data_grapher_volumned(paths, isplot=False, isCluster=False, augmentation=(),
                                seeds=np.random.normal(0, 0.02, 5000), isDataGen=True):
    # utility directory to save the pyplots
    radarData_path, videoData_path, mergedImg_path, out_path, identity_string = paths

    radar_3dscatter_path = 'F:/indexPen/figures/utils/radar_3dscatter'

    radar_data = list(pickle.load(open(radarData_path, 'rb')).items())
    radar_data.sort(key=lambda x: x[0])  # sort by timestamp
    videoData_list = os.listdir(videoData_path)
    videoData_timestamps = list(map(lambda x: float(x.strip('.jpg')), videoData_list))

    style.use('fivethirtyeight')
    white_color = 'rgb(255, 255, 255)'
    black_color = 'rgb(0, 0, 0)'
    red_color = 'rgb(255, 0, 0)'

    DBSCAN_esp = 0.2
    DBSCAN_minSamples = 3

    # input data for the classifier that has the shape n*4*100, n being the number of samples
    num_padding = 100
    data_for_classifier = np.zeros((len(radar_data), num_padding, 4))
    data_for_classifier_flattened = np.zeros(
        (len(radar_data), 4 * num_padding + 1 + 1 + 1))  # + 1 + 1 for the timestamp as integer ratio

    fnt = ImageFont.truetype("arial.ttf", 16)

    # Retrieve the first timestamp
    starting_timestamp = radar_data[0][0]

    interval_index = 1

    # removed and recreate the merged image folder
    if isplot:
        if os.path.isdir(mergedImg_path):
            shutil.rmtree(mergedImg_path)
        os.mkdir(mergedImg_path)

    volume_shape = (25, 25, 25)

    interval_volume_list = []
    volumes_for_this_interval = []

    interval_sec = 5
    sample_per_sec = 20
    sample_per_interval = interval_sec * sample_per_sec

    aug_string = ''
    if augmentation:
        print('Use augmentation: ' + str(augmentation))
        for aug in augmentation:
            aug_string += '_' + aug
    else:
        print('No augmentation applied')

    print('Label Cheat-sheet:')
    print('0 for A')
    print('1 for D')
    print('2 for L')
    print('3 for M')
    print('4 for P')

    label_array = []

    num_write = 2
    this_label = 1.0

    for i, radarFrame in enumerate(radar_data):

        # retrieve the data
        timestamp, fData = radarFrame

        # calculate the interval
        if (timestamp - starting_timestamp) >= 5.0:
            num_intervaled_samples = len(volumes_for_this_interval)
            if num_intervaled_samples < sample_per_interval / 4:
                raise Exception('Not Enough Data Points, killed')

            # decide the label
            if num_write == 1:
                if interval_index % (5 * num_write) == 1:
                    this_label = 0
                elif interval_index % (5 * num_write) == 2:
                    this_label = 1  # for label D
                elif interval_index % (5 * num_write) == 3:
                    this_label = 2  # for label L
                elif interval_index % (5 * num_write) == 4:
                    this_label = 3  # for label M
                elif interval_index % (5 * num_write) == 0:
                    this_label = 4  # for label P
            elif num_write == 2:
                if interval_index % (5 * num_write) == 1 or interval_index % (5 * num_write) == 2:
                    this_label = 0
                elif interval_index % (5 * num_write) == 3 or interval_index % (5 * num_write) == 4:
                    this_label = 1  # for label D
                elif interval_index % (5 * num_write) == 5 or interval_index % (5 * num_write) == 6:
                    this_label = 2  # for label L
                elif interval_index % (5 * num_write) == 7 or interval_index % (5 * num_write) == 8:
                    this_label = 3  # for label M
                elif interval_index % (5 * num_write) == 9 or interval_index % (5 * num_write) == 0:
                    this_label = 4  # for label P
            label_array.append(this_label)  # for label A

            print('Label for the last interval is ' + str(this_label) + ' Num Samples: ' + str(
                len(volumes_for_this_interval)))
            print('')

            # add padding, pre-padded
            if len(volumes_for_this_interval) < sample_per_interval:
                while len(volumes_for_this_interval) < sample_per_interval:
                    volumes_for_this_interval.insert(0, np.expand_dims(np.zeros(volume_shape), axis=0))
            elif len(volumes_for_this_interval) > sample_per_interval:  # we take only the 75 most recent
                volumes_for_this_interval = volumes_for_this_interval[-75:]
            volumes_for_this_interval = np.asarray(volumes_for_this_interval)
            interval_volume_list.append(volumes_for_this_interval)
            volumes_for_this_interval = []
            # increment the timestamp and interval index
            starting_timestamp = starting_timestamp + 5.0
            interval_index = interval_index + 1
        # end of end of interval processing

        print('Processing ' + str(i + 1) + ' of ' + str(len(radar_data)) + ', interval = ' + str(interval_index))

        if isplot:
            mergedImg_path_intervaled = os.path.join(mergedImg_path, str(interval_index - 1))

            if not os.path.isdir(mergedImg_path_intervaled):
                os.mkdir(mergedImg_path_intervaled)

            closest_video_timestamp = min(videoData_timestamps,
                                          key=lambda x: abs(x - timestamp))
            closest_video_path = os.path.join(videoData_path, str(closest_video_timestamp) + '.jpg')
            closest_video_img = Image.open(closest_video_path)

            # plot the radar scatter
            ax1 = plt.subplot(2, 2, 1, projection='3d')
            ax1.set_xlim((-0.3, 0.3))
            ax1.set_ylim((-0.3, 0.3))
            ax1.set_zlim((-0.3, 0.3))
            ax1.set_xlabel('X', fontsize=10)
            ax1.set_ylabel('Y', fontsize=10)
            ax1.set_zlabel('Z', fontsize=10)
            ax1.set_title('Detected Points', fontsize=10)
            # plot the detected points
            ax1.scatter(fData['x'], fData['y'], fData['z'], c=fData['doppler'], marker='o')

        data = np.asarray([fData['x'], fData['y'], fData['z'], fData['doppler']]).transpose()
        # Do DBSCAN cluster ###########################################
        # map the points to their doppler value, this is for retrieving the doppler value after clustering
        if isCluster:
            doppler_dict = {}
            for point in data:
                doppler_dict[tuple(point[:3])] = point[3:]
            # get rid of the doppler for clustering TODO should we consider the doppler in clustering?
            data = data[:, :3]

            db = DBSCAN(eps=DBSCAN_esp, min_samples=DBSCAN_minSamples).fit(data)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            if isplot:
                ax2 = plt.subplot(2, 2, 2, projection='3d')
                ax2.set_xlim((-0.3, 0.3))
                ax2.set_ylim((-0.3, 0.3))
                ax2.set_zlim((-0.3, 0.3))
                ax2.set_xlabel('X', fontsize=10)
                ax2.set_ylabel('Y', fontsize=10)
                ax2.set_zlabel('Z', fontsize=10)
                ax2.set_title('Clustered Points', fontsize=10)

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
                if xyz.any():  # in case there are none objects
                    clusters.append(xyz)  # append this cluster data to the cluster list
                # each cluster is a 3 * n matrix
                xyz = data[class_member_mask & ~core_samples_mask]
                if isplot:
                    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', c=np.array([col]), s=12,
                                marker='X')  # plot the noise

            # find the center for each cluster
            clusters_centers = list(
                map(lambda xyz: np.array([np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])]), clusters))
            clusters.sort(key=lambda xyz: distance.euclidean((0.0, 0.0, 0.0), np.array(
                [np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])])))

            # plot the clusters
            for xyz, col in zip(clusters, colors):
                if isplot:
                    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', c=np.array([col]), s=28,
                                marker='o')  # plot the cluster points

            #############################
            # clear the hand cluster
            hand_cluster = []
            bbox = (0.2, 0.2, 0.2)

            if len(clusters) > 0:
                hand_cluster = clusters[0]
                point_num = hand_cluster.shape[0]

                # if the cluster is outside the 20*20*20 cm bounding box
                distance_from_center = distance.euclidean((0.0, 0.0, 0.0), np.array(
                    [np.mean(hand_cluster[:, 0]), np.mean(hand_cluster[:, 1]), np.mean(hand_cluster[:, 2])]))

                if distance_from_center > distance.euclidean((0.0, 0.0, 0.0),
                                                             bbox):  # if the core of the cluster is too far away from the center
                    hand_cluster = np.zeros((hand_cluster.shape[0], hand_cluster.shape[1] + 1))
                else:
                    doppler_array = np.zeros((point_num, 1))
                    for j in range(point_num):
                        doppler_array[j:, ] = doppler_dict[tuple(hand_cluster[j, :3])]
                    # append back the doppler
                    hand_cluster = np.append(hand_cluster, doppler_array, 1)
        else:
            hand_cluster = data

        hand_cluster = np.array(hand_cluster)

        # apply augmentation to hand cluster #############################
        if hand_cluster.size != 0:
            # apply augmentations
            if 'trans' in augmentation:
                for p in np.nditer(hand_cluster[:, :3], op_flags=['readwrite']):
                    p[...] = p + random.choice(seeds)
            if 'rot' in augmentation:
                hand_cluster[:, :3] = rotateX(hand_cluster[:, :3], 720 * random.choice(seeds))
                hand_cluster[:, :3] = rotateY(hand_cluster[:, :3], 720 * random.choice(seeds))
                hand_cluster[:, :3] = rotateZ(hand_cluster[:, :3], 720 * random.choice(seeds))
            if 'scale' in augmentation:
                s = 1 + random.choice(seeds)
                hand_cluster[:, :3] = scale(hand_cluster[:, :3], x=s, y=s, z=s)

            if isplot:
                ax3 = plt.subplot(2, 2, 3, projection='3d')
                ax3.set_xlim((-0.3, 0.3))
                ax3.set_ylim((-0.3, 0.3))
                ax3.set_zlim((-0.3, 0.3))
                ax3.set_xlabel('X', fontsize=10)
                ax3.set_ylabel('Y', fontsize=10)
                ax3.set_zlabel('Z', fontsize=10)
                ax3.set_title('Hand Cluster', fontsize=10)

                ax3.scatter(hand_cluster[:, 0], hand_cluster[:, 1], hand_cluster[:, 2], 'o', c=hand_cluster[:, 3], s=28,
                            marker='o')

        # create 3D feature space #############################
        frame_3D_volume = snapPointsToVolume(hand_cluster, volume_shape, isClipping=('clipping' in augmentation))
        volumes_for_this_interval.append(np.expand_dims(frame_3D_volume, axis=0))

        # Plot the hand cluster #########################################

        #################################################################
        # Combine the three images
        if isplot:
            plt.savefig(os.path.join(radar_3dscatter_path, str(timestamp) + '.jpg'))
            radar_3dscatter_img = Image.open(os.path.join(radar_3dscatter_path, str(timestamp) + '.jpg'))

            images = [closest_video_img, radar_3dscatter_img]  # add image here to arrange them horizontally
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
            (x, y) = (20, 10)
            message = "Timestamp Difference, abs(rt-vt): " + str(timestamp_difference)
            draw.text((x, y), message, fill=white_color, font=fnt)
            # draw the timestamp
            (x, y) = (20, 30)
            message = "Timestamp: " + str(timestamp)
            draw.text((x, y), message, fill=white_color, font=fnt)

            # draw the number of points
            (x, y) = (20, 60)
            message = "Number of detected points: " + str(xyz.shape[0])
            draw.text((x, y), message, fill=white_color, font=fnt)

            # draw the number of clusters and number of noise point on the clutter plot
            if isCluster:
                (x, y) = (20, 80)
                message = "Number of clusters: " + str(n_clusters_)
                draw.text((x, y), message, fill=white_color, font=fnt)
                (x, y) = (20, 100)
                message = "Number of outliers: " + str(n_noise_)
                draw.text((x, y), message, fill=white_color, font=fnt)

            # save the combined image
            new_im.save(
                os.path.join(mergedImg_path_intervaled, str(timestamp) + '_' + str(timestamp.as_integer_ratio()[0]) +
                             '_' + str(timestamp.as_integer_ratio()[1]) + '_' + str(interval_index) + '.jpg'))
            plt.close('all')

    # process the last interval ##########################################################################
    if len(volumes_for_this_interval) <= 100:
        num_intervaled_samples = len(volumes_for_this_interval)
        if num_intervaled_samples < sample_per_interval / 4:
            print('Not Enough Data Points, saving')
        else:
            # decide the label
            if num_write == 1:
                if interval_index % (5 * num_write) == 1:
                    this_label = 0
                elif interval_index % (5 * num_write) == 2:
                    this_label = 1  # for label D
                elif interval_index % (5 * num_write) == 3:
                    this_label = 2  # for label L
                elif interval_index % (5 * num_write) == 4:
                    this_label = 3  # for label M
                elif interval_index % (5 * num_write) == 0:
                    this_label = 4  # for label P
            elif num_write == 2:
                if interval_index % (5 * num_write) == 1 or interval_index % (5 * num_write) == 2:
                    this_label = 0
                elif interval_index % (5 * num_write) == 3 or interval_index % (5 * num_write) == 4:
                    this_label = 1  # for label D
                elif interval_index % (5 * num_write) == 5 or interval_index % (5 * num_write) == 6:
                    this_label = 2  # for label L
                elif interval_index % (5 * num_write) == 7 or interval_index % (5 * num_write) == 8:
                    this_label = 3  # for label M
                elif interval_index % (5 * num_write) == 9 or interval_index % (5 * num_write) == 0:
                    this_label = 4  # for label P
            label_array.append(this_label)  # for label A

            print('Label for the last interval is ' + str(this_label) + ' Num Samples: ' + str(
                len(volumes_for_this_interval)))
            print('')

            # add padding, pre-padded
            if len(volumes_for_this_interval) < sample_per_interval:
                while len(volumes_for_this_interval) < sample_per_interval:
                    volumes_for_this_interval.insert(0, np.expand_dims(np.zeros(volume_shape), axis=0))
            elif len(volumes_for_this_interval) > sample_per_interval:  # we take only the 75 most recent
                volumes_for_this_interval = volumes_for_this_interval[-75:]
            volumes_for_this_interval = np.asarray(volumes_for_this_interval)
            interval_volume_list.append(volumes_for_this_interval)
            volumes_for_this_interval = []
            # increment the timestamp and interval index
            starting_timestamp = starting_timestamp + 5.0
            interval_index = interval_index + 1

    # start of post processing ##########################################################################
    label_array = np.asarray(label_array)
    interval_volume_array = np.asarray(interval_volume_list)

    # validate the output shapes
    assert interval_volume_array.shape == (50, 100, 1) + volume_shape
    assert len(label_array) == 50

    print('Saving csv and npy to ' + out_path + '...')
    if isDataGen:
        dataset_path = 'F:/indexPen/dataset'
        label_dict_path = 'F:/indexPen/labels/label_dict.p'
        # load label dict
        if os.path.exists(label_dict_path):
            label_dict = pickle.load(open(label_dict_path, 'rb'))
        else:  # create anew if does not exist
            label_dict = {}

        # put the label into the dict
        for l_index, l in enumerate(label_array):
            label_dict[identity_string + str(l_index) + aug_string] = l
        # save label dict to disk
        pickle.dump(label_dict, open(label_dict_path, 'wb'))

        # save the data chunks (intervaled volumns)
        for d_index, d in enumerate(interval_volume_array):
            print('Saving chunk #' + str(d_index))
            np.save(os.path.join(dataset_path, identity_string + str(d_index) + aug_string), d)

    else:
        np.save(os.path.join(out_path, 'label_array'), label_array)
        np.save(os.path.join(out_path, 'intervaled_3D_volumes_' + str(volume_shape[0]) + 'x' + aug_string),
                interval_volume_array)

    print('Done saving to ' + out_path)


def generate_path(subject_name: str, case_index: int, mode='indexPen') -> tuple:

    identity_string = subject_name + '_' + str(case_index)
    f_dir = 'f_data_' + identity_string
    v_dir = 'v_data_' + identity_string

    dataRootPath = 'F:/' + mode + '/data'
    figureRootPath = 'F:/' + mode + '/figures'

    radarData_path = os.path.join(dataRootPath, f_dir, 'f_data.p')
    videoData_path = os.path.join(dataRootPath, v_dir, 'cam2')
    mergedImg_path = os.path.join(figureRootPath, identity_string)
    out_path = os.path.join('F:/' + mode + '/csv_augmented', identity_string)

    return radarData_path, videoData_path, mergedImg_path, out_path, identity_string

def generate_train_val_ids(test_ratio, dataset_path='D:/indexPen/dataset'):
    data_ids = os.listdir(dataset_path)

    data_ids = list(map(lambda x: os.path.splitext(x)[0], data_ids))

    random.shuffle(data_ids)
    num_data = len(data_ids)

    line = int((1-test_ratio) * num_data)

    train_ids = data_ids[:line]
    test_ids = data_ids[line:]

    data_dict = {'train': [], 'validation': []}

    for train_sample in train_ids:
        data_dict['train'].append(train_sample)

    for test_sample in test_ids:
        data_dict['validation'].append((test_sample))

    return data_dict


def radar_data_grapher_volumned_track(paths, isPlot=False, isCluster=False, augmentation=(),
                                      seeds=np.random.normal(0, 0.02, 5000)):
    # utility directory to save the pyplots
    radarData_path, videoData_path, mergedImg_path, out_path, identity_string = paths

    radar_3dscatter_path = 'F:/thumouse/figures/utils/'
    dataset_path = 'F:/thumouse/dataset/'

    radar_data = list(pickle.load(open(radarData_path, 'rb')).items())
    radar_data.sort(key=lambda x: x[0])  # sort by timestamp
    videoData_list = os.listdir(videoData_path)
    videoData_timestamps = list(map(lambda x: float(x.strip('.jpg')), videoData_list))

    style.use('fivethirtyeight')
    white_color = 'rgb(255, 255, 255)'

    DBSCAN_esp = 0.2
    DBSCAN_minSamples = 3

    # input data for the classifier that has the shape n*4*100, n being the number of samples
    num_padding = 100

    fnt = ImageFont.truetype("arial.ttf", 16)

    # removed and recreate the merged image folder
    if isPlot:
        if os.path.isdir(mergedImg_path):
            shutil.rmtree(mergedImg_path)
        os.mkdir(mergedImg_path)

    volume_shape = (25, 25, 25)

    circular_vol_buffer = []

    interval_sec = 5
    sample_per_sec = 20
    sample_per_interval = interval_sec * sample_per_sec

    aug_string = ''
    if augmentation:
        print('Use augmentation: ' + str(augmentation))
        for aug in augmentation:
            aug_string += '_' + aug
    else:
        print('No augmentation applied')

    for i, radarFrame in enumerate(radar_data):

        # retrieve the data
        timestamp, fData = radarFrame

        if isPlot:

            closest_video_timestamp = min(videoData_timestamps,
                                          key=lambda x: abs(x - timestamp))
            closest_video_path = os.path.join(videoData_path, str(closest_video_timestamp) + '.jpg')
            closest_video_img = Image.open(closest_video_path)

            # plot the radar scatter
            ax1 = plt.subplot(2, 2, 1, projection='3d')
            ax1.set_xlim((-0.3, 0.3))
            ax1.set_ylim((-0.3, 0.3))
            ax1.set_zlim((-0.3, 0.3))
            ax1.set_xlabel('X', fontsize=10)
            ax1.set_ylabel('Y', fontsize=10)
            ax1.set_zlabel('Z', fontsize=10)
            ax1.set_title('Detected Points', fontsize=10)
            # plot the detected points
            ax1.scatter(fData['x'], fData['y'], fData['z'], c=fData['doppler'], marker='o')

        data = np.asarray([fData['x'], fData['y'], fData['z'], fData['doppler']]).transpose()
        # Do DBSCAN cluster ###########################################
        # map the points to their doppler value, this is for retrieving the doppler value after clustering
        if isCluster:
            doppler_dict = {}
            for point in data:
                doppler_dict[tuple(point[:3])] = point[3:]
            # get rid of the doppler for clustering TODO should we consider the doppler in clustering?
            data = data[:, :3]

            db = DBSCAN(eps=DBSCAN_esp, min_samples=DBSCAN_minSamples).fit(data)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            if isPlot:
                ax2 = plt.subplot(2, 2, 2, projection='3d')
                ax2.set_xlim((-0.3, 0.3))
                ax2.set_ylim((-0.3, 0.3))
                ax2.set_zlim((-0.3, 0.3))
                ax2.set_xlabel('X', fontsize=10)
                ax2.set_ylabel('Y', fontsize=10)
                ax2.set_zlabel('Z', fontsize=10)
                ax2.set_title('Clustered Points', fontsize=10)

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
                if xyz.any():  # in case there are none objects
                    clusters.append(xyz)  # append this cluster data to the cluster list
                # each cluster is a 3 * n matrix
                xyz = data[class_member_mask & ~core_samples_mask]
                if isPlot:
                    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', c=np.array([col]), s=12,
                                marker='X')  # plot the noise

            # find the center for each cluster
            clusters_centers = list(
                map(lambda xyz: np.array([np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])]), clusters))
            clusters.sort(key=lambda xyz: distance.euclidean((0.0, 0.0, 0.0), np.array(
                [np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])])))

            # plot the clusters
            for xyz, col in zip(clusters, colors):
                if isPlot:
                    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', c=np.array([col]), s=28,
                                marker='o')  # plot the cluster points

            #############################
            # clear the hand cluster
            hand_cluster = []
            bbox = (0.2, 0.2, 0.2)

            if len(clusters) > 0:
                hand_cluster = clusters[0]
                point_num = hand_cluster.shape[0]

                # if the cluster is outside the 20*20*20 cm bounding box
                distance_from_center = distance.euclidean((0.0, 0.0, 0.0), np.array(
                    [np.mean(hand_cluster[:, 0]), np.mean(hand_cluster[:, 1]), np.mean(hand_cluster[:, 2])]))

                if distance_from_center > distance.euclidean((0.0, 0.0, 0.0),
                                                             bbox):  # if the core of the cluster is too far away from the center
                    hand_cluster = np.zeros((hand_cluster.shape[0], hand_cluster.shape[1] + 1))
                else:
                    doppler_array = np.zeros((point_num, 1))
                    for j in range(point_num):
                        doppler_array[j:, ] = doppler_dict[tuple(hand_cluster[j, :3])]
                    # append back the doppler
                    hand_cluster = np.append(hand_cluster, doppler_array, 1)
        else:
            hand_cluster = data

        hand_cluster = np.array(hand_cluster)

        # apply augmentation to hand cluster #############################
        if hand_cluster.size != 0:
            # apply augmentations
            if 'trans' in augmentation:
                for p in np.nditer(hand_cluster[:, :3], op_flags=['readwrite']):
                    p[...] = p + random.choice(seeds)
            if 'rot' in augmentation:
                hand_cluster[:, :3] = rotateX(hand_cluster[:, :3], 720 * random.choice(seeds))
                hand_cluster[:, :3] = rotateY(hand_cluster[:, :3], 720 * random.choice(seeds))
                hand_cluster[:, :3] = rotateZ(hand_cluster[:, :3], 720 * random.choice(seeds))
            if 'scale' in augmentation:
                s = 1 + random.choice(seeds)
                hand_cluster[:, :3] = scale(hand_cluster[:, :3], x=s, y=s, z=s)

            if isPlot:
                ax3 = plt.subplot(2, 2, 3, projection='3d')
                ax3.set_xlim((-0.3, 0.3))
                ax3.set_ylim((-0.3, 0.3))
                ax3.set_zlim((-0.3, 0.3))
                ax3.set_xlabel('X', fontsize=10)
                ax3.set_ylabel('Y', fontsize=10)
                ax3.set_zlabel('Z', fontsize=10)
                ax3.set_title('Hand Cluster', fontsize=10)

                ax3.scatter(hand_cluster[:, 0], hand_cluster[:, 1], hand_cluster[:, 2], 'o', c=hand_cluster[:, 3], s=28,
                            marker='o')

        # create 3D feature space #############################
        frame_3D_volume = snapPointsToVolume(hand_cluster, volume_shape, isClipping=('clipping' in augmentation))
        circular_vol_buffer.append(np.expand_dims(frame_3D_volume, axis=0))

        print('Processing ' + str(i + 1) + ' of ' + str(len(radar_data)) + ' Circular buffer size: ' + str(len(circular_vol_buffer)))

        if len(circular_vol_buffer) == 75:
            # save this sequence
            print('saving npy...', end='')
            this_path = os.path.join(dataset_path, str(timestamp.as_integer_ratio()[0]) + '_' + str(timestamp.as_integer_ratio()[1]))
            if os.path.exists(this_path):
                raise Exception('File ' + this_path + ' already exists. THIS SHOULD NEVER HAPPEN!')
            np.save(this_path, circular_vol_buffer)
            print('saved to ' + this_path)
            circular_vol_buffer = circular_vol_buffer[1:]
        elif len(circular_vol_buffer) > 75:
            raise Exception('Circular Buffer Overflows. THIS SHOULD NEVER HAPPEN!')

        # Plot the hand cluster #########################################

        #################################################################
        # Combine the three images
        if isPlot:
            plt.savefig(os.path.join(radar_3dscatter_path, str(timestamp) + '.jpg'))
            radar_3dscatter_img = Image.open(os.path.join(radar_3dscatter_path, str(timestamp) + '.jpg'))

            images = [closest_video_img, radar_3dscatter_img]  # add image here to arrange them horizontally
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
            (x, y) = (20, 10)
            message = "Timestamp Difference, abs(rt-vt): " + str(timestamp_difference)
            draw.text((x, y), message, fill=white_color, font=fnt)
            # draw the timestamp
            (x, y) = (20, 30)
            message = "Timestamp: " + str(timestamp)
            draw.text((x, y), message, fill=white_color, font=fnt)

            # draw the number of points
            (x, y) = (20, 60)
            message = "Number of detected points: " + str(xyz.shape[0])
            draw.text((x, y), message, fill=white_color, font=fnt)

            # draw the number of clusters and number of noise point on the clutter plot
            if isCluster:
                (x, y) = (20, 80)
                message = "Number of clusters: " + str(n_clusters_)
                draw.text((x, y), message, fill=white_color, font=fnt)
                (x, y) = (20, 100)
                message = "Number of outliers: " + str(n_noise_)
                draw.text((x, y), message, fill=white_color, font=fnt)

            # save the combined image
            new_im.save(
                os.path.join(mergedImg_path, str(timestamp) + '_' + str(timestamp.as_integer_ratio()[0]) +
                             '_' + str(timestamp.as_integer_ratio()[1]) + '.jpg'))
            plt.close('all')