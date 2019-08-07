import pickle

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

import time

def preprocess_frame(frame):
    """

    :param frame: np array with input shape (n, 4)
    :return hand cluster of shape (200)
    """
    DBSCAN_esp = 0.2
    DBSCAN_minSamples = 4
    num_padding = 50

    clusters = []
    doppler_dict = {}

    output_shape = (200)

    for point in frame:
        doppler_dict[tuple(point[:3])] = point[3:]
    frame = frame[:, :3]

    db = DBSCAN(eps=DBSCAN_esp, min_samples=DBSCAN_minSamples).fit(frame)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)

    for k in unique_labels:
        if k == -1 and len(unique_labels) == 1:  # only noise is present
            return np.zeros(output_shape)
        class_member_mask = (labels == k)
        xyz = frame[class_member_mask & core_samples_mask]

        if xyz.any():  # in case there are none objects
            clusters.append(xyz)

    # find the center for each cluster
    clusters_centers = list(
        map(lambda xyz: np.array([np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])]), clusters))
    clusters.sort(key=lambda xyz: distance.euclidean((0.0, 0.0, 0.0), np.array(
        [np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])])))

    if len(clusters) > 0:
        hand_cluster = clusters[0]

        if len(hand_cluster) < DBSCAN_minSamples: # if this is just the noise cluster
            return np.zeros(output_shape)

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
        if point_num > num_padding:
            hand_cluster_padded = hand_cluster[:, :50]  # we take only the first 50 points
        else:
            hand_cluster_padded = np.pad(hand_cluster, ((0, num_padding - point_num), (0, 0)), 'constant',
                                     constant_values=0)
    else:
        hand_cluster_padded = np.zeros(output_shape)

    return hand_cluster_padded.reshape(output_shape)

frameArray = np.load('F:/test_frameArray.npy')
start = time.time()
result = preprocess_frame(frameArray[2])
end = time.time()
print('Preprocessing frame took ' + str(end-start))