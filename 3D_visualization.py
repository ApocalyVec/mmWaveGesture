import matplotlib
import PyQt5
# matplotlib.use('Qt5Agg')  # using interactive backend

import matplotlib.pyplot as plt
plt.ion()  # turn on interactive mode

import matplotlib.animation as animation
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D  # must import this for fig's projection 3D to work!

import pickle

style.use('fivethirtyeight')

data = pickle.load(open('data/072319_02/f_data.p', 'rb'))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

frame_data = data[1563935364.4086826]

ax.scatter(frame_data['x'], frame_data['y'], frame_data['z'], c=frame_data['doppler'], marker='D')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()