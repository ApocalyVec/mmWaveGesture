import _thread
import collections
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
detObj = {}
frameData = {}
draw_x_y = None
draw_z_v = None
# variables for prediction
data_q = collections.deque(maxlen=None)
pred_thread_stop_flag = False
# thread related variables
pred_stop_flag = Event()

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


# Main loop


def load_model(model_path, encoder_path=None):
    model = NeuralNetwork()
    model.load(file_name=model_path)

    if encoder_path is not None:
        encoder = pickle.load(encoder_path)
        return model, encoder
    else:
        return model



class PredictionThread(Thread):
    def __init__(self, thread_id, model, timestep):
        Thread.__init__(self)
        self.thread_id = thread_id
        self.model = model
        self.timestep = timestep

    def run(self):
        global pred_thread_stop_flag

        while not pred_thread_stop_flag:
            pred_result = pred_func(model=self.model, timestep=self.timestep)
            if pred_result is not None:
                print('result is ' + str(pred_result))


class InputThread(Thread):
    def __init__(self, thread_id):
        Thread.__init__(self)
        self.thread_id = thread_id

    def run(self):
        global pred_stop_flag

        input()
        pred_stop_flag.set()


def pred_func(model: NeuralNetwork, timestep):
    global data_q

    if len(data_q) > timestep:
        data = np.expand_dims(np.asarray([data_q.pop() for _ in range(timestep)]), axis=0)
        return model.predict(x=data)
    else:
        print('Not Enough Data to predict')
        return None



def main():
    global data_q, frameData, dataOk
    global pred_thread_stop_flag
    # gui related globals
    global draw_x_y, draw_z_v
    # thread related globals
    global pred_stop_flag

    today = datetime.datetime.now()

    # root_dn = 'data/f_data-' + str(today).replace(':', '-').replace(' ', '_')

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

    print("Started, input anything in this console and hit enter to stop")

    # start the prediction thread

    thread1 = PredictionThread(1, model=load_model('D:/thumouse/trained_models/thuMouse_noAug_349e-3.h5'), timestep=10)
    thread2 = InputThread(2)

    thread1.start()
    thread2.start()

    while True:
        dataOk, detObj = update()

        if dataOk:
            # Store the current frame into frameData
            frameData[time.time()] = detObj
            frameRow = np.asarray([detObj['x'], detObj['y'], detObj['z'], detObj['doppler']]).transpose()
            data_q.append(np.expand_dims(preprocess_frame(frameRow, isClipping=False), axis=0))

            time.sleep(0.033)  # This is framing frequency Sampling frequency of 30 Hz

        if pred_stop_flag.is_set():
            # set the stop flag for threads
            pred_thread_stop_flag = True
            # release sem for threads so that they can stop

            # wait for other threads to finish
            thread1.join()
            thread2.join()

            # close the plot window
            win.close()

            break
    # Stop sequence
    print('Stopped')

if __name__ == '__main__':
    main()
