import _thread
import collections
import math
import msvcrt
import pickle
import random
from threading import Thread
import threading

from tkinter import *
import tkinter as tk

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
from utils.iwr1443_utils import parseConfigFile, readAndParseData14xx
from sklearn.preprocessing import MinMaxScaler

is_simulate = False

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
pred_stop_flag = threading.Event()

max_timestep = 20
data_shape = (1, 25, 25, 25)

# IWR1443 Interface Globals------------------------------------------------------------------------------------------
dataOk = False
CLIport = {}
Dataport = {}
configFileName = 'D:/code/DoubleMU/1443config.cfg'
configParameters = None

# IWR1443 Interface Functions------------------------------------------------------------------------------------------
def serialConfig(configFileName):
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports

    # Raspberry pi
    # CLIport = serial.Serial('/dev/ttyACM0', 115200)
    # Dataport = serial.Serial('/dev/ttyACM1', 921600)

    # For WINDOWS, CHANGE those serial port to match your machine's configuration
    CLIport = serial.Serial('COM5', 115200)
    Dataport = serial.Serial('COM4', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i + '\n').encode())
        print(i)
        time.sleep(0.01)

    return CLIport, Dataport


def simulateData():
    detObj = dict()

    num_points = random.randint(1, 40)
    detObj['x'] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))
    detObj['y'] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))
    detObj['z'] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))
    detObj['doppler'] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))

    time.sleep(0.033)

    return True, 0, detObj


def update():
    global dataOk
    global detObj
    global is_simulate
    x = []
    y = []
    z = []
    doppler = []

    # Read and parse the received data
    if is_simulate:
        dataOk, frameNumber, detObj = simulateData()
    else:
        dataOk, frameNumber, detObj = readAndParseData14xx(Dataport, configParameters)

    if dataOk:
        x = -detObj['x']
        y = detObj['y']
        z = detObj['z']
        doppler = detObj['doppler']

    # update the plot
    draw_x_y.setData(x, y)
    draw_z_v.setData(z, doppler)

    return dataOk, detObj


# Main loop


def load_model(model_path, encoder_path=None):
    model = NeuralNetwork()
    model.load(file_name=model_path)

    if encoder_path is not None:
        encoder = pickle.load(open(encoder_path, 'rb'))
        return model, encoder
    else:
        return model



class PredictionThread(Thread):
    def __init__(self, thread_id, model_encoder, timestep, gui_handle=None):

        Thread.__init__(self)
        self.thread_id = thread_id
        self.model, self.encoder = model_encoder
        self.timestep = timestep
        self.ring_buffer = np.zeros(tuple([timestep] + list(data_shape)))
        self.buffer_head = 0

        if gui_handle is not None:
            self.gui_handle = gui_handle
            self.x = 0
            self.y = 0


    def run(self):
        global pred_thread_stop_flag

        while not pred_thread_stop_flag:
            # retrieve the data from deque
            if len(data_q) != 0:
                self.ring_buffer[self.buffer_head] = data_q.pop()
                self.buffer_head += 1
                if self.buffer_head >= self.timestep:
                    self.buffer_head = 0

            # print('Head is at ' + str(self.buffer_head))
            pred_result = pred_func(model=self.model, data=np.expand_dims(self.ring_buffer, axis=0))  # expand dim for single sample batch
            decoded_result = self.encoder.inverse_transform(pred_result)

            self.x = min(max(self.x + decoded_result[0][0], -100), 100)
            self.y = min(max(self.y + decoded_result[0][1], -100), 100)

            print(str(self.x) + ' ' + str(self.y))
            print(str([decoded_result[0][0]]) + str([decoded_result[0][1]]))
            if self.gui_handle is not None:
                self.gui_handle.setData([self.x], [self.y])



class InputThread(Thread):
    def __init__(self, thread_id):
        Thread.__init__(self)
        self.thread_id = thread_id

    def run(self):
        global pred_stop_flag

        input()
        pred_stop_flag.set()


def pred_func(model: NeuralNetwork, data):
    return model.predict(x=data)

def main(is_simulate):
    global data_q, frameData, dataOk
    global pred_thread_stop_flag
    global max_timestep
    # gui related globals
    global draw_x_y, draw_z_v
    # thread related globals
    global pred_stop_flag
    # IWR1443 related Globals
    global CLIport, Dataport, configFileNameconfig, configParameters


    today = datetime.datetime.now()

    # root_dn = 'data/f_data-' + str(today).replace(':', '-').replace(' ', '_')

    warnings.simplefilter('ignore', np.RankWarning)

    # START QtAPPfor the plot
    app = QtGui.QApplication([])

    # Set the plot
    pg.setConfigOption('background', 'w')
    window = pg.GraphicsWindow(title="2D scatter plot")
    fig_z_y = window.addPlot()
    fig_z_y.setXRange(-0.5, 0.5)
    fig_z_y.setYRange(0, 1.5)
    fig_z_y.setLabel('left', text='Y position (m)')
    fig_z_y.setLabel('bottom', text='X position (m)')
    draw_x_y = fig_z_y.plot([], [], pen=None, symbol='o')
    fig_z_v = window.addPlot()
    fig_z_v.setXRange(-1, 1)
    fig_z_v.setYRange(-1, 1)
    fig_z_v.setLabel('left', text='Z position (m)')
    fig_z_v.setLabel('bottom', text='Doppler (m/s)')
    draw_z_v = fig_z_v.plot([], [], pen=None, symbol='o')

    # create thumouse window
    thumouse_gui = window.addPlot()
    thumouse_gui.setXRange(-100, 100)
    thumouse_gui.setYRange(-100, 100)
    draw_thumouse_gui = thumouse_gui.plot([], [], pen=None, symbol='o')


    print("Started, input anything in this console and hit enter to stop")

    # start the prediction thread
    thread1 = PredictionThread(1, model_encoder=load_model('D:/thumouse/trained_models/thuMouse_noAug_349e-3.h5',
                                                   encoder_path='D:/thumouse/scaler/mmScaler.p'), timestep=10, gui_handle = draw_thumouse_gui)
    thread2 = InputThread(2)

    thread1.start()
    thread2.start()

    if not is_simulate:
        # start IWR1443 serial connection
        CLIport, Dataport = serialConfig(configFileName)
        configParameters = parseConfigFile(configFileName)

    while True:
        dataOk, detObj = update()

        if dataOk:
            # Store the current frame into frameData
            frameData[time.time()] = detObj
            frameRow = np.asarray([detObj['x'], detObj['y'], detObj['z'], detObj['doppler']]).transpose()
            data_q.append(np.expand_dims(preprocess_frame(frameRow, isClipping=False), axis=0))  # expand dim for single channeled data

            time.sleep(0.033)  # This is framing frequency Sampling frequency of 30 Hz
        QtGui.QApplication.processEvents()
        if pred_stop_flag.is_set():
            # set the stop flag for threads
            pred_thread_stop_flag = True
            # populate data queue with dummy data so that the prediction thread can stop
            (data_q.append(np.zeros(data_shape))  for _ in range(max_timestep))
            if not is_simulate:
                # close serial interface
                CLIport.write(('sensorStop\n').encode())
                CLIport.close()
                Dataport.close()

            # wait for other threads to finish
            thread1.join()
            thread2.join()

            # close the plot window
            window.close()

            break
    # Stop sequence
    print('Stopped')

if __name__ == '__main__':
    main(is_simulate=is_simulate)
