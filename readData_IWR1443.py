import _thread
import pickle

import serial
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

import datetime
import os
import time

import warnings

from iwr1443_utils import readAndParseData14xx, parseConfigFile

configFileName = '1443config.cfg'

CLIport = {}
Dataport = {}

"""
# global to hold the average x, y, z of detected points

those are for testing swipe gestures
note that for now, only x and y are used

those values is updated in the function: update
"""
x = []
y = []
z = []
doppler = []
# features = []

# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
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


# ------------------------------------------------------------------



# ------------------------------------------------------------------

# Funtion to update the data and display in the plot
def update():
    dataOk = 0
    global detObj

    x = []
    y = []
    z = []
    doppler = []

    # Read and parse the received data
    dataOk, frameNumber, detObj = readAndParseData14xx(Dataport, configParameters)

    if dataOk:
        # print(detObj)
        x = -detObj["x"]
        y = detObj["y"]
        z = detObj["z"]
        doppler = detObj["doppler"]  # doppler values for the detected points hopefully in m/s

    s_original.setData(x, y)
    s_processed.setData(z, doppler)

    QtGui.QApplication.processEvents()

    return dataOk


# -------------------------    MAIN   -----------------------------------------
today = datetime.datetime.now()
today = datetime.datetime.now()

root_dn = 'f_data-' + str(today).replace(':', '-').replace(' ', '_')
os.mkdir(root_dn)

warnings.simplefilter('ignore', np.RankWarning)
# Configurate the serial port
CLIport, Dataport = serialConfig(configFileName)

# Get the configuration parameters from the configuration file
configParameters = parseConfigFile(configFileName)

# START QtAPPfor the plot
app = QtGui.QApplication([])

# Set the plot 
pg.setConfigOption('background', 'w')
win = pg.GraphicsWindow(title="2D scatter plot")
p_original = win.addPlot()
p_original.setXRange(-0.5, 0.5)
p_original.setYRange(0, 1.5)
p_original.setLabel('left', text='Y position (m)')
p_original.setLabel('bottom', text='X position (m)')
s_original = p_original.plot([], [], pen=None, symbol='o')

# set the processed plot
p_processed = win.addPlot()
p_processed.setXRange(-1, 1)
p_processed.setYRange(-1, 1)
p_processed.setLabel('left', text='Z position (m)')
p_processed.setLabel('bottom', text='Doppler (m/s)')
s_processed = p_processed.plot([], [], pen=None, symbol='o')

# Main loop
detObj = {}
frameData = {}

def input_thread(a_list):
    input()
    interrupt_list.append(True)


interrupt_list = []
_thread.start_new_thread(input_thread, (interrupt_list,))


while True:
    try:
        # Update the data and check if the data is okay
        dataOk = update()

        if dataOk:
            # Store the current frame into frameData
            # frameData[currentIndex] = detObj
            frameData[time.time()] = detObj
            # features.append([detObj['x'],detObj['y'],detObj['z'],detObj['doppler']])

        time.sleep(0.033)  # Additional Comment: this is framing frequency Sampling frequency of 30 Hz

        if interrupt_list:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()
            win.close()
            print("Exiting")

            # save radar frame data
            file_path = os.path.join(root_dn, 'f_data.p')
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(frameData, pickle_file)
            break

    # Stop the program and close everything if Ctrl + c is pressed

    except KeyboardInterrupt:
        CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        Dataport.close()
        win.close()

        # save radar frame data
        file_path = os.path.join(root_dn, 'f_data.p')
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(frameData, pickle_file)

        break
