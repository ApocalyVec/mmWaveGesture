import numpy as np
import math
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import pandas as pd

# FROM https://www.geeksforgeeks.org/linear-regression-python-implementation/
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

def generate_single_plot(plot_save_path, mergedImg_path, font, closest_video_img, closest_video_timestamp, xyz, n_clusters_, n_noise_, timestamp, cluster):

    white_color = 'rgb(255, 255, 255)'

    cluster_flattened = cluster.reshape(( -1))
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
def generate_plot(radar_data, videoData_timestamps, videoData_path, DBSCAN_esp, DBSCAN_minSamples, num_padding, font, radar_3dscatter_path, mergedImg_path):
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

        data_for_classifier_flattened[i] = generate_single_plot(radar_3dscatter_path, mergedImg_path, font, closest_video_img, closest_video_timestamp, xyz,
            n_clusters_, n_noise_, timestamp, hand_cluster_padded)
    return data_for_classifier_flattened

def label(folder_path, data_file):
    img_folder = os.listdir(folder_path)
    gesture_timestamp = dict()
    for gesture in img_folder:
        if not gesture == '.DS_Store':
            gesture_timestamp[gesture[0:1]] = os.listdir(os.path.join(folder_path, gesture))
            gesture_timestamp[gesture[0:1]].remove('.DS_Store')
            gesture_timestamp[gesture[0:1]] = list(map(lambda x: float(x.strip('.jpg')), gesture_timestamp[gesture[0:1]]))

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
                #raise ValueError('found a timestamp that is not in the given csv file!')

    for timestamp in range(len(data)):
        print('labeling ' + str(timestamp) + ' of ' + str(len(data)))
        for gesture in gesture_timestamp:
            if data.loc[timestamp].iat[1] in gesture_timestamp[gesture]:
                data.loc[timestamp].iat[0] = float(gesture)
                break
    return (data, not_found)

# volumn.shape = (5, 5, 5)
def addPointToVolumn(x, y, z, v, radius, volume):
    """
    make sure volume is a square
    :param x: scale -1 to 1
    :param y:
    :param z:
    :param heat:
    :param volume:
    """
    pass
