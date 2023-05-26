# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:03:39 2023

@author: Antonin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:02:23 2023

@author: Administrator
"""

# Grab_MultipleCameras.cpp
# ============================================================================
# This sample illustrates how to grab and process images from multiple cameras
# using the CInstantCameraArray class. The CInstantCameraArray class represents
# an array of instant camera objects. It provides almost the same interface
# as the instant camera for grabbing.
# The main purpose of the CInstantCameraArray is to simplify waiting for images and
# camera events of multiple cameras in one thread. This is done by providing a single
# RetrieveResult method for all cameras in the array.
# Alternatively, the grabbing can be started using the internal grab loop threads
# of all cameras in the CInstantCameraArray. The grabbed images can then be processed by one or more
# image event handlers. Please note that this is not shown in this example.
# ============================================================================

import os
import cv2

os.environ["PYLON_CAMEMU"] = "3"

from pypylon import genicam
from pypylon import pylon
import sys
from datetime import datetime
import shutil

#%% PARAMETRES
nom_exp = "test_marina"
date = "230307"
Calibration = True
# Number of images to be grabbed.
countOfImagesToGrab = 3000

pathH = "D:\\refref6\\"
# if Calibration :
#     pathH += "\\references\\"

# Limits the amount of cameras used for grabbing.
# It is important to manage the available bandwidth when grabbing with multiple cameras.
# This applies, for instance, if two GigE cameras are connected to the same network adapter via a switch.
# To manage the bandwidth, the GevSCPD interpacket delay parameter and the GevSCFTD transmission delay
# parameter can be set for each GigE camera device.
# The "Controlling Packet Transmission Timing with the Interpacket and Frame Transmission Delays on Basler GigE Vision Cameras"
# Application Notes (AW000649xx000)
# provide more information about this topic.
# The bandwidth used by a FireWire camera device can be limited by adjusting the packet size.
maxCamerasToUse = 2

# The exit code of the sample application.
exitCode = 0

exp_time = 7000

facq = 10

gain = 1

# camera.MaxNumBuffer = 15






#%% MAIN

try:

    # Get the transport layer factory.
    tlFactory = pylon.TlFactory.GetInstance()

    # Get all attached devices and exit application if no device is found.
    devices = tlFactory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

    l = cameras.GetSize()

    # Create and attach all Pylon Devices.
    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[i]))
        cam.Open()
        cam.Gain.SetValue(gain)
        cam.ExposureTime.SetValue(exp_time)
        cam.PixelFormat.SetValue("RGB8")
        cam.AcquisitionFrameRateEnable.SetValue(True);
        cam.AcquisitionFrameRate.SetValue(facq);
        # Print the model name of the camera.
        print("Using device ", cam.GetDeviceInfo().GetModelName())

    # Starts grabbing for all cameras starting with index 0. The grabbing
    # is started for one camera after the other. That's why the images of all
    # cameras are not taken at the same time.
    # However, a hardware trigger setup can be used to cause all cameras to grab images synchronously.
    # According to their default configuration, the cameras are
    # set up for free-running continuous acquisition.
    cameras.StartGrabbing()

    # Grab c_countOfImagesToGrab from the cameras.
    for i in range(countOfImagesToGrab):
        if not cameras.IsGrabbing():
            break

        cameras.Gain = 3
        # cameras.ExposureTime.SetValue(exp_time)

        grabResult = cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # When the cameras in the array are created the camera context value
        # is set to the index of the camera in the array.
        # The camera context is a user settable value.
        # This value is attached to each grab result and can be used
        # to determine the camera that produced the grab result.
        cameraContextValue = grabResult.GetCameraContext()
        
        # Print the index and the model name of the camera.
        print("Camera ", cameraContextValue, ": ", cameras[cameraContextValue].GetDeviceInfo().GetModelName())

        # Now, the image data can be processed.
        print("GrabSucceeded: ", grabResult.GrabSucceeded())
        print("SizeX: ", grabResult.GetWidth())
        print("SizeY: ", grabResult.GetHeight())
        img = grabResult.GetArray()
        print("Gray value of first pixel: ", img[0, 0])
        info_img = datetime.now()
        info_img = str(info_img)
        info_img = info_img[:4] + info_img[5:7] + info_img[8:10]+ info_img[11:13] + info_img[14:16]+info_img[17:19] +info_img[20:25]
        pathHH = os.path.join(pathH, info_img + '.tiff')
        
        # Display the resulting frame
        cv2.imwrite(pathHH, img)
        # cv2.imshow("img " + str(i), img)

except genicam.GenericException as e:
    # Error handling
    print("An exception occurred.", e.GetDescription())
    exitCode = 1

print(cameras[0].ExposureTime.GetValue())
print(cameras[1].ExposureTime.GetValue())

# Comment the following two lines to disable waiting on exit.
cameras.Close()
sys.exit(exitCode)

#%% CLOSE

cameras.Close()
sys.exit(exitCode)

#%% TRI IMAGES
pathH = "H:\Canada\Data\Stereo\d230309\\tie_16\\references_3\\"
if Calibration :
    liste_photos = os.listdir(pathH)
    os.mkdir(pathH + "\\cam1")
    os.mkdir(pathH + "\\cam2")
    u= 0
    for i in liste_photos :
        if (u % 2) == 0 :
            shutil.copy(pathH + i,pathH + "\\cam1\\" + i)
        if (u % 2) == 1 :
            shutil.copy(pathH + i,pathH + "\\cam2\\" + i)
        u += 1
        if (u % 100) == 0 :
            print(u)
        






