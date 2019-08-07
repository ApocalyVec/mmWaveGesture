import pickle
import numpy as np
from minisom import MiniSom
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pylab import bone, pcolor, colorbar, plot, show
from sklearn.preprocessing import MinMaxScaler

# import the raw data

# data_path_list = ['F:/onNotOn_raw/zl_onNoton_raw_flattened.p',
#                   'F:/onNotOn_raw/ag_onNoton_raw_flattened.p',
#                   'F:/onNotOn_raw/zy_onNoton_raw_flattened.p']

data_path_list = ['F:/onNotOn_raw/zl_onNoton_raw.p',
                  'F:/onNotOn_raw/ag_onNoton_raw.p',
                  'F:/onNotOn_raw/zy_onNoton_raw.p']

data_list = list(map(lambda path: pickle.load(open(path, 'rb')), data_path_list))

# concatenate the data vertically
dataset = np.concatenate(data_list, axis=0)

# Without the Timestamp ###############################
X = dataset.reshape((len(dataset),
                     200,))  # Change 200 to 201 depending on whether the data has timestamp (pre-flattened with test_radar_data_labeler)
som = MiniSom(x=100, y=100, input_len=200, sigma=1.0, learning_rate=0.5)

# With the Timestamp ##########################
# min-max normalize the timestamp,
# X = dataset.reshape((len(dataset), 201,))  # Change 200 to 201 depending on whether the data has timestamp (pre-flattened with test_radar_data_labeler)
# timestamp_scaler = MinMaxScaler(feature_range=(0, 1))
# timestamp_col_scaled = timestamp_scaler.fit_transform(X[:, 0].reshape(-1, 1))
# X[:, 0] = timestamp_col_scaled.reshape((len(timestamp_col_scaled)))
# som = MiniSom(x=50, y=50, input_len=201, sigma=1.0, learning_rate=0.5)

# som.random_weights_init(X)
som.train_random(data=X, num_iteration=1000)

# visualize results
label_path = 'F:\config_detection\labels/labeled_onNotOn_080719.csv'
label_array = pd.read_csv(label_path).values[:, 1:]
label_col = label_array[:, 0]

figure(num=None, figsize=(10, 10), dpi=150, facecolor='w', edgecolor='k')
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['x', 'v', '^', '1', '2']
colors = ['y', 'r', 'b', 'm', 'g']

for i, x in enumerate(label_array):
    w = som.winner(x[4:])  # the value before the forth column are timestamps
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[int(label_col[i])-1],
         markeredgecolor=colors[int(label_col[i])-1],
         markerfacecolor='None',
         markersize=5,
         markeredgewidth=2)

show()
