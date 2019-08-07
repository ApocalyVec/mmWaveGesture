import pandas as pd
import numpy as np
import pickle

from keras.layers import Dropout
from sklearn.preprocessing import OneHotEncoder

label_path = 'F:\config_detection\labels/labeled_onNotOn_080719.csv'
label_df = pd.read_csv(label_path)

X = label_df.iloc[:, 5:]
Y = label_df.iloc[:, 1]
Y = np.asarray(Y)
Y = np.expand_dims(Y, axis=1)
onehotencoder = OneHotEncoder(categories='auto')
Y = onehotencoder.fit_transform(Y).toarray()

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# As a general practice: the number of neurons in each layer is ave(input_dim + output_dim), in this case,
# number of neurons = (11 + 1)/2 = 6
classifier.add(Dense(units=200, kernel_initializer='uniform', activation='relu', input_dim=200))
classifier.add(Dropout(p=0.1))
# Adding the second hidden layer
classifier.add(Dense(units=200, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))
# Adding the output hidden layer
# units = 1 for output_dim = 1
classifier.add(Dense(units=5, kernel_initializer='uniform', activation='softmax'))

# Compiling the ANN
adam = optimizers.adam(lr=0.0005, clipnorm=1.)  # use half the learning rate as adam optimizer default
classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set - the actual training
history = classifier.fit(X, Y, batch_size=1, epochs=100)

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# plt.savefig('F:/config_detection/models/onNotOn_ANN/history_080719_2')
classifier.save('F:/config_detection/models/onNotOn_ANN/classifier_080719_2')
pickle.dump(onehotencoder, open('F:/config_detection/models/onNotOn_ANN/encoder_080719_2', 'wb'))