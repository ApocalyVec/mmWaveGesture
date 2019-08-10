import numpy as np
import pandas as pd
import pickle
import os

# Building the model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers

# get the data
from sklearn.preprocessing import OneHotEncoder

input_dir_list = ['D:/zy_1/']

x_train = None
y_train = None
for input_dir in input_dir_list:
    data = np.load(os.path.join(input_dir, 'intervaled_ts_removed.npy'))
    label_array = np.load(os.path.join(input_dir, 'label_array.npy'))

    # put in the data
    if x_train is None:
        x_train = data
    else:
        x_train = np.concatenate((x_train, data))

    if y_train is None:
        y_train = label_array
    else:
        y_train = np.concatenate((y_train, label_array))

# Onehot encode Y ############################################
onehotencoder = OneHotEncoder(categories='auto')
y_train = onehotencoder.fit_transform(np.expand_dims(y_train, axis=1)).toarray()

# Build the RNN ###############################################
interval_sec = 5
sample_per_sec = 15
sample_per_interval = interval_sec * sample_per_sec
# Initialising the RNN
regressiveClassifier = Sequential()

# batch size = 3337, num_timestep = 100,
regressiveClassifier.add(LSTM(units=400, return_sequences=True,  input_shape=(sample_per_interval, 400)))
regressiveClassifier.add(Dropout(rate=0.2))

regressiveClassifier.add(LSTM(units=400, return_sequences=True))
regressiveClassifier.add(Dropout(rate=0.2))

regressiveClassifier.add(LSTM(units=400, return_sequences=True))
regressiveClassifier.add(Dropout(0.2))

regressiveClassifier.add(LSTM(units=400, return_sequences=False))  # or return_sequences is default at False
regressiveClassifier.add(Dropout(rate=0.2))

# dense layer
# regressiveClassifier.add(Dense(units=400, kernel_initializer='uniform', activation='relu'))
# regressiveClassifier.add(Dropout(rate=0.1))
#
# regressiveClassifier.add(Dense(units=400, kernel_initializer='uniform', activation='relu'))
# regressiveClassifier.add(Dropout(rate=0.1))

regressiveClassifier.add(Dense(5, activation='softmax'))

adam_lr5e_4 = optimizers.adam(lr=5e-5, clipnorm=1.)  # use half the learning rate as adam optimizer default
regressiveClassifier.compile(optimizer=adam_lr5e_4, loss='categorical_crossentropy', metrics=['accuracy'])

history = regressiveClassifier.fit(x_train, y_train, epochs=50000, batch_size=10)

# plot train history
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# save train result
# date = '080819'
# regressiveClassifier.save(os.path.join('F:/palmpad/models', date + 'classifier.h5'))
# pickle.dump(onehotencoder, open(os.path.join('F:/palmpad/models', date + 'encoder.p', 'wb'), 'wb'))

# TODO change learning rate