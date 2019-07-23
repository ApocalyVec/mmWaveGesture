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

configFileName = '1443config.cfg'

CLIport = {}
Dataport = {}

"""
# global to hold the average x, y, z of detected points

those are for testing swipe gestures
note that for now, only x and y are used

those values is updated in the function: update
"""
avg_x = 0.0
avg_y = 0.0
avg_z = 0.0
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

    # TODO Windows, NEED to CHANGE those serial port to match your machine's configuration
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

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:

        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 3

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(splitWords[2])
            idleTime = int(splitWords[3])
            rampEndTime = int(splitWords[5]);
            freqSlopeConst = int(splitWords[8]);
            numAdcSamples = int(splitWords[10]);
            numAdcSamplesRoundTo2 = 1;

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2;

            digOutSampleRate = int(splitWords[11]);

        # Get the information about the frame configuration    
        elif "frameCfg" in splitWords[0]:

            chirpStartIdx = int(splitWords[1]);
            chirpEndIdx = int(splitWords[2]);
            numLoops = int(splitWords[3]);
            numFrames = int(splitWords[4]);
            framePeriodicity = int(splitWords[5]);

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
            2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
            2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (
            2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    return configParameters


# ------------------------------------------------------------------

# Funtion to read and parse the incoming data
def readAndParseData14xx(Dataport, configParameters):
    # global byteBuffer, byteBufferLength
    byteBuffer = np.zeros(2 ** 15, dtype='uint8')
    byteBufferLength = 0;

    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12;
    BYTE_VEC_ACC_MAX_SIZE = 2 ** 15;
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1;
    MMWDEMO_UART_MSG_RANGE_PROFILE = 2;
    maxBufferSize = 2 ** 15;
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    # Initialize variables
    magicOK = 0  # Checks if magic number has been read
    dataOK = 0  # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}

    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    byteCount = len(byteVec)

    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount

    # Check that the buffer has some data
    if byteBufferLength > 16:

        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteVec == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteVec[loc:loc + 8]
            # if check == magicWord:
            #     startIdx.append(loc)
            if np.all(check == magicWord):
                startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:

            # Remove the data before the first start index
            if startIdx[0] > 0:
                byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBufferLength = byteBufferLength - startIdx[0]

            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)

            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

        # Initialize the pointer index
        idX = 0

        # Read the header
        magicNumber = byteBuffer[idX:idX + 8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4

        # Read the TLV messages
        for tlvIdx in range(numTLVs):

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Check the header of the TLV message
            tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4

            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:

                # word array to convert 4 bytes to a 16 bit number
                word = [1, 2 ** 8]
                tlv_numObj = np.matmul(byteBuffer[idX:idX + 2], word)
                idX += 2
                tlv_xyzQFormat = 2 ** np.matmul(byteBuffer[idX:idX + 2], word)
                idX += 2

                # Initialize the arrays
                rangeIdx = np.zeros(tlv_numObj, dtype='int16')
                dopplerIdx = np.zeros(tlv_numObj, dtype='int16')
                peakVal = np.zeros(tlv_numObj, dtype='int16')
                x = np.zeros(tlv_numObj, dtype='int16')
                y = np.zeros(tlv_numObj, dtype='int16')
                z = np.zeros(tlv_numObj, dtype='int16')

                for objectNum in range(tlv_numObj):
                    # Read the data for each object
                    rangeIdx[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    dopplerIdx[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    peakVal[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    x[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    y[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    z[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2

                # Make the necessary corrections and calculate the rest of the data
                rangeVal = rangeIdx * configParameters["rangeIdxToMeters"]
                dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"] / 2 - 1)] = dopplerIdx[dopplerIdx > (
                        configParameters["numDopplerBins"] / 2 - 1)] - 65535
                dopplerVal = dopplerIdx * configParameters["dopplerResolutionMps"]
                # x[x > 32767] = x[x > 32767] - 65536
                # y[y > 32767] = y[y > 32767] - 65536
                # z[z > 32767] = z[z > 32767] - 65536
                x = x / tlv_xyzQFormat
                y = y / tlv_xyzQFormat
                z = z / tlv_xyzQFormat

                # Store the data in the detObj dictionary
                detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx, \
                          "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z}

                dataOK = 1

                # print(detObj['range'].mean())

            elif tlv_type == MMWDEMO_UART_MSG_RANGE_PROFILE:
                idX += tlv_length

        # Remove already processed data
        if idX > 0 and dataOK == 1:
            shiftSize = idX

            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

    return dataOK, frameNumber, detObj


# ------------------------------------------------------------------

# Funtion to update the data and display in the plot
def update():
    dataOk = 0
    global detObj

    global x
    global y
    global z
    global doppler

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


    """
    # test for gesture detection?
    
        # get the new x,y average if there are detected points
    # if len(x) == 0:
    #     new_avg_x = 0
    # else:
    #     new_avg_x = statistics.mean(x)

    # if len(y) == 0:
    #     new_avg_y = 0
    # else:
    #     new_avg_y = statistics.mean(y)

    # factor = int(100*(new_avg_x - avg_x))
    # print(str(factor))  # + ", " + str(new_avg_y - avg_y))

    # update the old average
    # avg_x = new_avg_x
    # avg_y = new_avg_y

    # set data for the original 2d scatter plot

    x_disp = 0
    y_disp = 0
    fit_x_line = []
    fit_y_line = []
    x_ol_removed = []
    y_ol_removed = []
    closest_point = (0, 0, 0)
    if len(x) != 0:
        # x_disp, y_disp, fit_x_line, fit_y_line = process_2d_pcd(x, y, z, doppler)
        closest_point = process_2d_pcd(x, y, z, doppler)

    # s_processed.setData([x_disp], [y_disp])
    """

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


# animating scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax = fig.gca(projection='3d')
# ax = fig.gca(projection='3d')
# plt.ion()
# plt.show()
# ani = animation.FuncAnimation(fig, animate, interval=1000)

# fig = plt.gcf()
# fig.show()
# fig.canvas.draw()
# plt.show()

def input_thread(a_list):
    input()
    interrupt_list.append(True)


interrupt_list = []
_thread.start_new_thread(input_thread, (interrupt_list,))


while True:
    try:
        # Update the data and check if the data is okay
        dataOk = update()

        # ax.clear()
        # ax.set_xdata(x)
        # ax.set_ydata(y)

        # ax.scatter(x, y, z, alpha=0.8, edgecolors='none', s=30)
        # plt.draw()
        #
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # plt.plot([1], [2])
        # fig.canvas.draw()

        if dataOk:
            # Store the current frame into frameData
            # frameData[currentIndex] = detObj
            frameData[time.time()] = detObj
            # features.append([detObj['x'],detObj['y'],detObj['z'],detObj['doppler']])
        # print(x)
        # print(y)
        # print(z)
        # print(doppler)

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
