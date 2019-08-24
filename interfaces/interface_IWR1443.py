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
from matplotlib import pyplot as plt

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
general_thread_stop_flag = False
# thread related variables
main_stop_event = threading.Event()

max_timestep = 20
data_shape = (1, 25, 25, 25)

# GUI related------------------------------------------------------------------------------------------
window = None
thm_gui_size = 640, 480

# IWR1443 Interface Globals------------------------------------------------------------------------------------------
dataOk = False
CLIport = {}
Dataport = {}
configFileName = 'D:/code/DoubleMU/1443config.cfg'
configParameters = None

# Model Globals
# my_mode = ['thm', 'idp']
my_mode = ['idp']
# idp_classify_threshold = 0.83730006  # avg of wrong mean and correct mean
idp_classify_threshold = 0.7

thm_model_path = 'D:/code/DoubleMU/models/thuMouse_model.h5'
thm_scaler_path = 'D:/code/DoubleMU/models/scalers/thm_scaler.p'

idp_model_path = 'D:/code/DoubleMU/models/palmPad_model.h5'


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


class onehot_decoder():

    def inverse_transform(datum):
        return np.argmax(datum)


def load_model(model_path, encoder=None):
    model = NeuralNetwork()
    model.load(file_name=model_path)

    if encoder is not None:
        if type(encoder) == str:
            encoder = pickle.load(open(encoder, 'rb'))
            return model, encoder
        elif type(encoder) == onehot_decoder:
            return model, encoder
    else:
        return model


x_list = []
y_list = []

class PredictionThread(Thread):
    def __init__(self, thread_id, model_encoder_dict, timestep, thumouse_gui=None, mode='thm'):

        Thread.__init__(self)
        self.thread_id = thread_id
        self.model_encoder_dict = model_encoder_dict
        self.timestep = timestep
        # create a sequence buffer of shape: timestemp * shape of the data
        self.mode = mode
        if 'thm' in mode:
            self.thumouse_gui = thumouse_gui

        if 'idp' in mode:
            pass

    def run(self):
        global general_thread_stop_flag
        global thm_gui_size

        global idp_classify_threshold

        idp_threshold = idp_classify_threshold  # copy as a local variable

        gui_wid_hei = thm_gui_size

        # maX = StreamingMovingAverage(window_size=10)
        # maY = StreamingMovingAverage(window_size=10)

        mouse_x = 0
        mouse_y = 0
        thm_x_factor = 15
        thm_y_factor = 5

        #TODO
        buffer_size = 35
        sequence_buffer = np.zeros(tuple([buffer_size] + list(data_shape)))

        idp_pred_dict = {0: 'A', 1: 'D', 2: 'L', 3: 'M', 4: 'P', 5: 'nothing'}

        while not general_thread_stop_flag:
            # retrieve the data from deque
            if len(data_q) != 0:
                # ditch the tail, append to head
                sequence_buffer = np.concatenate((sequence_buffer[1:], np.expand_dims(data_q.pop(), axis=0)))

                if 'idp' in self.mode:
                    time.sleep(0.5)
                    idp_pre_result = pred_func(model=self.model_encoder_dict['idp'][0], data=np.expand_dims(np.concatenate((sequence_buffer, np.zeros((65, 1, 25, 25, 25)))), axis=0))[0]
                    pre_argmax = np.argmax(idp_pre_result)
                    pre_amax = np.amax(idp_pre_result)

                    if pre_amax > idp_threshold:  # a character is written
                        if pre_argmax == 5:
                            print('No One is Writing' + '    amax = ' + str(pre_amax))
                        else:
                            print('You just wrote: ' + idp_pred_dict[pre_argmax] + '    amax = ' + str(pre_amax))
                            # clear the buffer
                            sequence_buffer = np.zeros(tuple([buffer_size] + list(data_shape)))
                    else:
                        print('No writing, amax = ' + str(pre_amax))

                if 'thm' in self.mode:
                    thm_pred_result = pred_func(model=self.model_encoder_dict['thm'][0],
                                            data=np.expand_dims(sequence_buffer[-1],  # always take the head
                                                                axis=0))  # expand dim for single sample batch
                    decoded_result = self.model_encoder_dict['thm'][1].inverse_transform(thm_pred_result)

                    delta_x = decoded_result[0][0] * thm_x_factor
                    delta_y = decoded_result[0][1] * thm_y_factor

                    # avg_x = maX.process(delta_x)
                    # avg_y = maY.process(delta_y)

                    mouse_x = min(max(mouse_x + delta_x, 0), gui_wid_hei[0])
                    mouse_y = min(max(mouse_y + delta_y, 0), gui_wid_hei[1])

                    x_list.append(delta_x)
                    y_list.append(delta_y)

                    if self.thumouse_gui is not None:
                        self.thumouse_gui.setData([mouse_x], [mouse_y])

                # print(str(delta_x) + ' ' + str(delta_y))
                # print(str([decoded_result[0][0]]) + str([decoded_result[0][1]]))


class StreamingMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)


class InputThread(Thread):
    def __init__(self, thread_id):
        Thread.__init__(self)
        self.thread_id = thread_id

    def run(self):
        global main_stop_event

        input()
        main_stop_event.set()


def pred_func(model: NeuralNetwork, data):
    return model.predict(x=data)


def main(is_simulate):
    global data_q, frameData, dataOk
    global general_thread_stop_flag
    global max_timestep
    # gui related globals
    global draw_x_y, draw_z_v
    # thread related globals
    global main_stop_event
    # IWR1443 related Globals
    global CLIport, Dataport, configParameters

    global thm_gui_size
    global window

    global thm_model_path

    global my_mode

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
    thumouse_gui.setXRange(0, thm_gui_size[0])
    thumouse_gui.setYRange(0, thm_gui_size[1])
    draw_thumouse_gui = thumouse_gui.plot([], [], pen=None, symbol='o')

    print("Started, input anything in this console and hit enter to stop")

    # start the prediction thread
    model_dict = {'thm': load_model(thm_model_path,
                                    encoder=thm_scaler_path),
                  'idp': load_model(idp_model_path,
                                    encoder=onehot_decoder())}

    thread1 = PredictionThread(1, model_encoder_dict=model_dict,
                               timestep=100, thumouse_gui=draw_thumouse_gui, mode=my_mode)

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
            data_q.append(np.expand_dims(preprocess_frame(frameRow, isCluster=False, isClipping=False), axis=0))  # expand dim for single channeled data

            QtGui.QApplication.processEvents()
        if main_stop_event.is_set():
            # set the stop flag for threads
            general_thread_stop_flag = True
            # populate data queue with dummy data so that the prediction thread can stop
            (data_q.append(np.zeros(data_shape)) for _ in range(max_timestep))
            if not is_simulate:
                # close serial interface
                CLIport.write(('sensorStop\n').encode())
                CLIport.close()
                Dataport.close()

            # wait for other threads to finish
            thread1.join()
            thread2.join()

            break
    # Stop sequence
    print('Stopped')


if __name__ == '__main__':
    main(is_simulate=is_simulate)
