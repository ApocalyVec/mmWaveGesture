import pickle

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

import time

from sklearn.preprocessing import MinMaxScaler

from utils.path_utils import snapPointsToVolume


volume_shape = (25, 25, 25)


def preprocess_frame(data, isCluster=True, isClipping=False):
    """

    :param frame: np array with input shape (n, 4)
    :return hand cluster of shape (200)
    """
    DBSCAN_esp = 0.2
    DBSCAN_minSamples = 3
    num_padding = 100

    bbox = (0.2, 0.2, 0.2)

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

        unique_labels = set(labels)
        clusters = []
        for k in zip(unique_labels):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xyz = data[class_member_mask & core_samples_mask]
            if xyz.any():  # in case there are none objects
                clusters.append(xyz)  # append this cluster data to the cluster list
            # each cluster is a 3 * n matrix
            xyz = data[class_member_mask & ~core_samples_mask]

        # find the center for each cluster
        clusters_centers = list(
            map(lambda xyz: np.array([np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])]), clusters))
        clusters.sort(key=lambda xyz: distance.euclidean((0.0, 0.0, 0.0), np.array(
            [np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])])))

        #############################
        hand_cluster = []
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
    frame_3D_volume = snapPointsToVolume(hand_cluster, volume_shape, isClipping=isClipping)

    return frame_3D_volume

# frameArray = np.load('F:/test_frameArray.npy')
# start = time.time()
# result = preprocess_frame(frameArray[2])
# end = time.time()
# print('Preprocessing frame took ' + str(end-start))