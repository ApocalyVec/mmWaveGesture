import pickle
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from pylab import bone, pcolor, colorbar, plot, show

# import the raw data

data_path_list = ['F:/onNotOn_raw/zl_onNoton_raw.p',
                  'F:/onNotOn_raw/ag_onNoton_raw.p',
                  'F:/onNotOn_raw/zy_onNoton_raw.p']

data_list = list(map(lambda path: pickle.load(open(path, 'rb')), data_path_list))

# concatenate the data vertically
dataset = np.concatenate(data_list, axis=0)
# flatten
X = dataset.reshape((len(dataset), 200,))

som = MiniSom(x=1000, y=1000, input_len=200, sigma=1.0, learning_rate=0.5)
# som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# visualize results
bone()
pcolor(som.distance_map().T)
colorbar()

show()