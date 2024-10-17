# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:23:56 2023

@author: Banquise
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
import os
from PIL import Image

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.files.file_management as fm
import baptiste.image_processing.image_processing
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools
import baptiste.math.RDD as rdd
import baptiste.math.fits as fits
import baptiste.image_processing.FSD as img

path = 'E:\\Nicolas\\d230725\\230725_facq0.100Hz_7.79g_80.Hz_5.360V\\'
images = os.listdir(path)[:-1]
facq = 1
mparpixel = 0.10111223458038422649 / 1000

#%% plot and load images



image_0 = cv2.imread (path + images[0], cv2.IMREAD_GRAYSCALE)

[nx,ny] = np.shape(image_0)
x = np.linspace(0,nx-1 * mparpixel,nx)
y = np.linspace(0,ny-1 * mparpixel,ny)

disp.figurejolie()
disp.joliplot('x','y',x, y, table = image_0)


image_1 = cv2.imread (path + images[-1], cv2.IMREAD_GRAYSCALE)


disp.figurejolie()
disp.joliplot('x','y',x, y, table = image_1)

#%% binary

scale = 255
sepwb = 160

#Contrast enhancement

I_min, I_max = np.min(image_0),np.max(image_0)
image_gray_0 = ((image_0-I_min) / (I_max-I_min) * scale)


image_binaire_0 = image_gray_0 < sepwb

image_binaire_0 = (image_binaire_0 * scale).astype(np.uint8)
    
image_binaire_0 = img.erodedilate(image_binaire_0, kernel_iteration = 1, kernel_size = 1)

image_binaire_0 = cv2.bitwise_not(image_binaire_0)

disp.figurejolie()
disp.joliplot('x','y',x, y, table = image_binaire_0)


I_min, I_max = np.min(image_1),np.max(image_1)
image_gray_1 = ((image_1-I_min) / (I_max-I_min) * scale)


image_binaire_1 = image_gray_1 < sepwb

image_binaire_1 = (image_binaire_1* scale).astype(np.uint8)
    
image_binaire_1 = img.erodedilate(image_binaire_1, kernel_iteration = 1, kernel_size = 1)

image_binaire_1 = cv2.bitwise_not(image_binaire_1)

disp.figurejolie()
disp.joliplot('x','y',x, y, table = image_binaire_1)