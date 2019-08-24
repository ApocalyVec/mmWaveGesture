import os
import shutil

from utils.path_utils import generate_path, radar_data_grapher_volumned_track

specimen_list = {generate_path('ag', 0, mode='thumouse'),
                 generate_path('zy', 0, mode='thumouse'),
                 generate_path('zl', 0, mode='thumouse')
                 }

# use data augmentation

dataset_path = 'F:/thumouse/dataset_timestep_1_noClp/'
if os.path.exists(dataset_path):
    print('Removing old data in ' + dataset_path)
    shutil.rmtree(dataset_path)
os.mkdir(dataset_path)

for i, path in enumerate(specimen_list):
    # generate orignial data
    print('Processing specimen #' + str(i) + '__________________________________')
    # radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['trans'])
    # radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['rot'])
    # radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['scale'])

    radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['trans'], dataset_path=dataset_path, timesteps=1)
    radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['rot'], dataset_path=dataset_path, timesteps=1)
    radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['scale'], dataset_path=dataset_path, timesteps=1)

    radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['trans', 'rot'], dataset_path=dataset_path, timesteps=1)
    radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['trans', 'scale'], dataset_path=dataset_path, timesteps=1)

    radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['rot', 'scale'], dataset_path=dataset_path, timesteps=1)

    radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['trans', 'rot', 'scale'], dataset_path=dataset_path, timesteps=1)



    # radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['clipping'], dataset_path=dataset_path, timesteps=1)
    # radar_data_grapher_volumned_track(path, isCluster=True, augmentation=['trans', 'clipping'], dataset_path=dataset_path, timesteps=1)




