import pandas as pd
import numpy as np
import os

radar_data_path = 'F:/config_detection/figures/onNotOn_csv/all_onNotOn.csv'
label_folder = 'F:/config_detection/labels/label080719'

class_folders = os.listdir(label_folder)
radar_array = pd.read_csv(radar_data_path).values[:, 1:]  # remove the index column

class_dict = {}

for cf in class_folders:
    class_category = cf.split('_')[0]  # the first element in the file name is the categorical label
    class_file_list = os.listdir(os.path.join(label_folder, cf))
    for fn in class_file_list:
        fn = fn.strip('.jpg')
        timestamp = (int(fn.split('_')[1]), int(fn.split('_')[2]))
        class_dict[timestamp] = class_category

radar_array_labeled = np.zeros((len(class_dict), radar_array.shape[1]))
i = 0
for row in radar_array:
    if (int(row[2]), int(row[3])) in class_dict.keys():
        radar_array_labeled[i] = row
        radar_array_labeled[i][0] = class_dict[(int(row[2]), int(row[3]))]  # put the label on
        i += 1