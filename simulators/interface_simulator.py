import _thread
import msvcrt
import pickle
import random
from threading import Thread, Event, Lock, Semaphore

import serial
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

import datetime
import os
import time

import warnings

from classes.model_wrapper import NeuralNetwork
from utils.data_utils import preprocess_frame

x = []
y = []
z = []
doppler = []

# variables for prediction
buffer_size = 100
ring_buffer = np.empty((buffer_size, 1, 25, 25, 25), dtype=np.float64)
buffer_head = 0

dataOk = False

def simulateData():
    detObj = dict()

    num_points = random.randint(1, 40)
    detObj['x'] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))
    detObj['y'] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))
    detObj['z'] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))
    detObj['doppler'] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))

    time.sleep(0.033)

    return True, detObj


def update():
    dataOk = 0
    global detObj

    x = []
    y = []
    z = []
    doppler = []

    # Read and parse the received data
    dataOk, detObj = simulateData()

    if dataOk:
        x = -detObj['x']
        y = detObj['y']
        z = detObj['z']
        doppler = detObj['doppler']

    # update the plot
    draw_x_y.setData(x, y)
    draw_z_v.setData(z, doppler)
    QtGui.QApplication.processEvents()

    return dataOk, detObj


# -------------------------    MAIN   -----------------------------------------
today = datetime.datetime.now()
today = datetime.datetime.now()

root_dn = 'data/f_data-' + str(today).replace(':', '-').replace(' ', '_')

warnings.simplefilter('ignore', np.RankWarning)

# START QtAPPfor the plot
app = QtGui.QApplication([])

# Set the plot
pg.setConfigOption('background', 'w')
win = pg.GraphicsWindow(title="2D scatter plot")
fig_z_y = win.addPlot()
fig_z_y.setXRange(-0.5, 0.5)
fig_z_y.setYRange(0, 1.5)
fig_z_y.setLabel('left', text='Y position (m)')
fig_z_y.setLabel('bottom', text='X position (m)')
draw_x_y = fig_z_y.plot([], [], pen=None, symbol='o')

# set the processed plot
fig_z_v = win.addPlot()
fig_z_v.setXRange(-1, 1)
fig_z_v.setYRange(-1, 1)
fig_z_v.setLabel('left', text='Z position (m)')
fig_z_v.setLabel('bottom', text='Doppler (m/s)')
draw_z_v = fig_z_v.plot([], [], pen=None, symbol='o')

# Main loop
detObj = {}
frameData = {}

buffer_sem = Semaphore(0)

def increment_buffer_head():
    global buffer_head

    buffer_head += 1
    if buffer_head >= buffer_size:  # reset buffer head position
        buffer_head = 0


class PredictionThread(Thread):
    def __init__(self, thread_id, event):
        Thread.__init__(self)
        self.thread_id = thread_id
        self.stopped = event

    def run(self):
        while True:
            buffer_sem.acquire()  # increment the semaphore
            print('Predicting')
            time.sleep(0.03)


print("Started, press CTRL+C in this console to stop")

# start the prediction thread
stopFlag = Event()
thread1 = PredictionThread(1, stopFlag)
thread1.start()

start_time = time.time()

while True:
    dataOk, detObj = update()

    if dataOk:
        # Store the current frame into frameData
        frameData[time.time()] = detObj
        frameRow = np.asarray([detObj['x'], detObj['y'], detObj['z'], detObj['doppler']]).transpose()
        ring_buffer[buffer_head] = preprocess_frame(frameRow)
        increment_buffer_head()
        buffer_sem.release()  # increment the semaphore
        time.sleep(0.033)  # This is framing frequency Sampling frequency of 30 Hz

        if msvcrt.kbhit():
            if ord(msvcrt.getch()) == 27:
                break

# Stop sequence
stopFlag.set()
