# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:02:12 2023

@author: Banquise
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
from scipy.signal import convolve2d
from scipy.signal import medfilt2d
from scipy import ndimage
import os
from PIL import Image
from matplotlib import image as img
%run parametres_FSD.py
%run Functions_FSD.py
%run display_lib_gld.py
%run bilinear.py

path_image = "D:\Banquise\Baptiste\Resultats_video\d230111\d230111_MPPB1_LAS_26sur082_facq101Hz_texp3137us_Tmot002_Vmot025_Hw11cm_tacq020s\image_sequence\\Basler_a2A1920-160ucBAS__40232066__20230111_181428149_11468.tiff"
path_image = "D:\Banquise\Baptiste\Resultats_video\d230116\\test_image_12bit_demosaic\\Basler_bayerRG.tiff"
path_test = "D:\Banquise\Baptiste\Resultats_video\d230116\\test_image_12bit_demosaic\\test1.tiff"

path_image_rgb = "D:\Banquise\Baptiste\Resultats_video\d230116\\test_image_12bit_demosaic\\Basler_rgb.tiff"
path_image_bayerRG = "D:\Banquise\Baptiste\Resultats_video\d230116\\test_image_12bit_demosaic\\Basler_bayerRG.tiff"
path_image_bayerRG_8 = "D:\Banquise\Baptiste\Resultats_video\d230116\\test_image_12bit_demosaic\\Basler_bayerRG_8bits.tiff"

image_test = cv2.imread(path_test)
# figurejolie()
# joliplot("","",(),(),image = image_test)

image = cv2.imread(path_image_bayerRG, cv2.IMREAD_UNCHANGED)
# image = cv2.imread(path_image_bayerRG,cv2.COLOR_BAYER_BG2GRAY)
# image= image / np.max(image)
image_rgb = cv2.imread(path_image_rgb)
image_bayerRG_8 = cv2.imread(path_image_bayerRG_8, cv2.IMREAD_COLOR)
# img_rgb = cv2.COLOR_BayerBG2BGR(image)
image_lala = cv2.imread(path_image,cv2.COLOR_BayerBG2BGR )

# figurejolie()
# joliplot("","",(),(),image = image)
# figurejolie()
# joliplot("","",(),(),image = image_rgb)
figurejolie()
joliplot("","",(),(),image = image_bayerRG_8)



# image = image_bayerRG_8

# figurejolie()
# joliplot("","",(),(),image = uuu)
# bgr = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)


img_pillow = Image.open(path_image_bayerRG)
img_mat = img.imread(path_image_bayerRG)

data = np.fromfile(path_image_bayerRG, np.uint16)
data = np.fromfile(path_image_rgb, np.uint8)
"""
width = 5760
height = 3600

# open(path_image, "rb") as rawimg :
# Read the packed 12bits as bytes - each 3 bytes applies 2 pixels
data = np.fromfile(path_image, np.uint8, width * height * 3//2)

data = data.astype(np.uint16)  # Cast the data to uint16 type


result = np.zeros(data.size*2//3, np.uint16)  # Initialize matrix for storing the pixels.
# 12 bits packing: ######## ######## ########
#                  | 8bits| | 4 | 4  |  8   |
#                  |  lsb | |msb|lsb |  msb |
#                  <-----------><----------->
#                     12 bits       12 bits
# data = np.delete(data,(-1,-2))
data = np.append(data,[0])
result = np.append(result,[0])
result[0::2] = ((data[1::3] & 15) << 8) | data[0::3]
result[1::2] = (data[1::3] >> 4) | (data[2::3] << 4)
bayer_im = np.reshape(result, (height, width))
# Apply Demosacing (COLOR_BAYER_BG2BGR gives the best result out of the 4 combinations).
bgr = cv2.cvtColor(bayer_im, cv2.COLOR_BAYER_BG2BGR)  # The result is BGR format with 16 bits per pixel and 12 bits range [0, 2^12-1].

# Show image for testing (multiply by 16 because imshow requires full uint16 range [0, 2^16-1]).
cv2.imshow('bgr', cv2.resize(bgr*16, [width//10, height//10]))
# cv2.waitKey()
# cv2.destroyAllWindows()
# Convert to uint8 before saving as JPEG (not part of the conversion).
colimg = np.round(bgr.astype(float) * (255/4095))
# cv2.imwrite("test.jpeg", colimg)


filename = path_image
bayer = np.fromfile(filename, dtype=np.uint8).reshape((-1,3))
bayer32 = np.dot(bayer.astype(np.uint32), [1,256,65536])

his = bayer32 >> 12                                             
los = bayer32 &  0xfff      
"""
#%%

# A helper function for merging three single-channel images into an RGB image
def combine_channels(ch1, ch2, ch3):
    return np.dstack((ch1, ch2, ch3))

def demosaic_simple_naive_rgb(img):
    s=img.shape
    img=img.astype('uint16')
    red=img*1
    blue=img*1
    green=img*1
    for i in range(s[0]):
        for j in range(s[1]):
            im1,ip1,jm1,jp1=(max(0,i-1),min(i+1,s[0]-1),max(0,j-1),min(j+1,s[1]-1))
            if(i%2==0 and j%2==0):
                red[i,j]=img[i,j]
                blue[i,j]=(img[im1,jm1]+img[ip1,jm1]+img[ip1,jp1]+img[im1,jp1])/4
                green[i,j]=(img[im1,j]+img[ip1,j]+img[i,jm1]+img[i,jp1])/4
            elif(i%2==1 and j%2==1):
                blue[i,j]=img[i,j]
                red[i,j]=(img[im1,jm1]+img[ip1,jm1]+img[ip1,jp1]+img[im1,jp1])/4
                green[i,j]=(img[im1,j]+img[ip1,j]+img[i,jm1]+img[i,jp1])/4
            else:
                green[i,j]=img[i,j]
                if(i%2==0):
                    red[i,j]=(img[i,jm1]+img[i,jp1])/2
                    blue[i,j]=(img[im1,j]+img[ip1,j])/2
                else:
                    blue[i,j]=(img[i,jm1]+img[i,jp1])/2
                    red[i,j]=(img[im1,j]+img[ip1,j])/2
    return (red,green,blue)

def demosaic_simple_naive(img):
    red,green,blue=demosaic_simple_naive_rgb(img)
    kek=np.ones(img.shape)
    return combine_channels(red,green,blue).astype('uint16')

# tests.append(tests[0][:40,:40])
uu = demosaic_simple_naive(image)
#%%
figurejolie()
plt.imshow(uu[:,:,:3]) #should show a color image 
figurejolie()
plt.imshow(uu[:,:,3:6])  
figurejolie()
plt.imshow(uu[:,:,6:9])
figurejolie()
plt.imshow(uu[:,:,6:9] - uu[:,:,3:6])
                                 

