# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:25:48 2023

@author: Banquise

code inspriré de :
    
https://stackoverflow.com/questions/73682222/debayering-16-bit-bayer-encoded-raw-images

et 

https://stackoverflow.com/questions/70515648/reading-and-saving-12bit-raw-bayer-image-using-opencv-python



"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

path_image_bayerRG = "D:\Banquise\Baptiste\Resultats_video\d230116\\test_image_12bit_demosaic\\Basler_bayerRG.tiff"
path_image_bayerRG_8 = "D:\Banquise\Baptiste\Resultats_video\d230116\\test_image_12bit_demosaic\\Basler_bayerRG_8bits.tiff"

file = path_image_bayerRG
height = 1200
width = 1920

lines = height
line_samples = width
hbands = 1 # on suppose

numpixels = height*width*hbands

hsample_bits = 16

f = open(file,'rb')
pixelbytes = numpixels*hsample_bits//8
data = f.read(pixelbytes)

img = np.frombuffer(data, np.uint16).reshape(lines, line_samples)


#pour 16 bits purs, à partir d'un tableau taille image
img = (img >> 8) + (img << 8)

data = np.fromfile(path_image_bayerRG, np.uint8, width * height * 3//2)

data = data.astype(np.uint16)  # Cast the data to uint16 type

#pour12 bits en 16, à partir de la ligne de data
img_12bits = np.zeros(data.size*2//3, np.uint16)  # Initialize matrix for storing the pixels.
# 12 bits packing: ######## ######## ########
#                  | 8bits| | 4 | 4  |  8   |
#                  |  lsb | |msb|lsb |  msb |
#                  <-----------><----------->
#                     12 bits       12 bits

# data = np.delete(data,(-1,-2))
# data = np.append(data,[0])
# result = np.append(result,[0])
img_12bits[0::2] = ((data[1::3] & 15) << 8) | data[0::3]
img_12bits[1::2] = (data[1::3] >> 4) | (data[2::3] << 4)


bayer_im_12bits = np.reshape(img_12bits, (height, width))


img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
bayer_im_12bits = cv2.cvtColor(bayer_im_12bits, cv2.COLOR_BAYER_RG2RGB)

# img = img / np.max(img)
# bayer_im_12bits = bayer_im_12bits / np.max(bayer_im_12bits)

cv2.imshow('bgr', cv2.resize(bayer_im_12bits*16, [width//10, height//10]))



cv2.imshow('img',img)

cv2.imshow('12 bits', bayer_im_12bits)
