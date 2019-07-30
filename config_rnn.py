import numpy as np
import pickle

# build the data set

frameData_fn = 'data/072319_02/f_data.p'
groundTruth_fn = 'data/072319_02/v_pred.p'

frameData = pickle.load(open(frameData_fn, 'rb'))
groundTruth = pickle.load(open(groundTruth_fn, 'rb'))  # CNN predictions

# data preprocessing

# prepare the ground truth
x = []  # independent variable: radar frame data
y = []  # dependent variable: prediction from video frames
xy_timestamp_diffs = []

num_padding = 100

for entry in list(frameData.items()):
    data_timestamp = entry[0]
    data = entry[1]

    # add padding to the detected points

    data = np.asarray([data['x'], data['y'], data['z'], data['doppler']]).transpose()

    if data.shape[0] > num_padding:
        raise Exception('Insufficient Padding')

    data = np.pad(data, ((0, num_padding - data.shape[0]), (0, 0)), 'constant', constant_values=0)

    data = data.reshape((400,))  # flatten

    closest_prediction_timestamp = min(list(groundTruth.keys()),
                                       key=lambda x: abs(x - data_timestamp))  # finds the closest binary prediction
    closest_prediction = groundTruth[closest_prediction_timestamp]

    x.append(data)
    y.append(closest_prediction)
    xy_timestamp_diffs.append(data_timestamp - closest_prediction_timestamp)

# alter the independent variable so that entry: contains previous n timesteps
timestep = 100

x_train = []
y_train = []
train_timestamp_diffs = []

for i in range(timestep, len(x)):
    x_train.append(x[i - timestep:i])
    y_train.append(y[i])

    # double checking the timestamp
    train_timestamp_diffs.append(xy_timestamp_diffs[i])

x_train = np.asarray(x_train)
y_train = np.asarray(list(map(lambda item: item[0][0], y_train)))


# Building the model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressiveClassifier = Sequential()

# batch size = 3337, num_timestep = 100,
regressiveClassifier.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 400)))
regressiveClassifier.add(Dropout(0.2))

regressiveClassifier.add(LSTM(units=50, return_sequences=True))
regressiveClassifier.add(Dropout(0.2))

regressiveClassifier.add(LSTM(units=50, return_sequences=False))
regressiveClassifier.add(Dropout(0.2))

# dense layer
# sigmoid for binary output
regressiveClassifier.add(Dense(1, activation='sigmoid'))

regressiveClassifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = regressiveClassifier.fit(x_train, y_train, epochs=100, batch_size=32)
