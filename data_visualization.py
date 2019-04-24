from mpl_toolkits.mplot3d import axes3d
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import matplotlib.animation as animation
from matplotlib import style
from pylab import *
import time


style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

def animate(data):
    x, y = data

    ax1.clear()
    ax1.plot(x, y)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()


def plot_3d_scatter(data):
    """

    :param data: tuple: (x_list, y_list, z_list)
    """

    fig = plt.figure()

    ax = plt.axes(facecolor="1.0")
    ax = fig.add_subplot(1, 1, 1)
    ax = fig.gca(projection='3d')

    x, y, z = data
    ax.scatter(x, y, z, alpha=0.8, edgecolors='none', s=30)

    plt.title('Test 3d scatter plot')
    plt.legend(loc=2)
    plt.show()


# def transpose_data(data_xyz):
#     """
#
#     :param data: tuple: (list, list, list)
#     """