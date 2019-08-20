from keras.engine.saving import load_model
import pickle
import numpy as np
import os

from utils.path_utils import generate_train_val_ids

regressor = load_model('F:/thumouse/trained_models/bestSoFar_thuMouse_CRNN2019-08-19_01-50-23.442673.h5')

dataset_path = 'F:/thumouse/dataset'
label_dict_path = 'F:/thumouse/labels/label_dict.p'

label_dict = pickle.load(open(label_dict_path, 'rb'))

partition = generate_train_val_ids(0.1, dataset_path=dataset_path)

# get the Y and X
# take only the first 100 for plotting
X_test = []
Y_test = []

for i, val_sample in enumerate(partition['train']):
    print('Reading ' + str(i) + ' of 100')
    if i < 100:
        X_test.append(np.load(os.path.join(dataset_path, val_sample + '.npy')))
        Y_test.append(label_dict[val_sample.split('_')[2] + '_' + val_sample.split('_')[3]])
    else:
        break
X_test = np.asarray(X_test)

# make the prediction
Y_predict = regressor.predict_on_batch(X_test[:10])

# plot the result


