# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:11:34 2023

@author: Banquise


Code trouvé sur :
    
https://stackoverflow.com/questions/73682222/debayering-16-bit-bayer-encoded-raw-images


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

path_image_bayerRG = "D:\Banquise\Baptiste\Resultats_video\d230116\\test_image_12bit_demosaic\\Basler_bayerRG.tiff"
path_image_rgb = "D:\Banquise\Baptiste\Resultats_video\d230116\\test_image_12bit_demosaic\\Basler_rgb.tiff"
image_path = "D:\Banquise\Baptiste\Resultats_video\d230116\\test_image_12bit_demosaic\\ZL0_0206_0685235537_613RAD_N0071836ZCAM08234_1100LMA02.IMG"
image_path = path_image_rgb


"""
hlines = lines, pas de diff avec ou sans le h
"""

def readHeader(file):
    # print("Calling readHeader")
    f = open(file,'rb')
    continuing = 1
    count = 0
    
    h_bytes = -1
    h_lines = -1
    h_line_samples = -1
    h_sample_type = 'UNSET' #MSB_INTEGER, IEEE_REAL
    h_sample_bits = -1
    h_bands = -1
    while continuing == 1:
        line = f.readline()
        count = count + 1
        arr = str(line, 'utf8').split("=")
        arr[0] = str(arr[0]).strip()
        if 'BYTES' == arr[0] and len(arr[0])>1:
            h_bytes=int(str(arr[1]).strip())
        elif 'LINES' == arr[0] and len(arr[0])>1: 
            h_lines=int(str(arr[1]).strip())
        elif 'LINE_SAMPLES' == arr[0] and len(arr[0])>1:
            h_line_samples=int(str(arr[1]).strip())
        elif 'SAMPLE_TYPE' == arr[0] and len(arr[0])>1:
            h_sample_type=str(arr[1]).strip()
        elif 'SAMPLE_BITS' == arr[0] and len(arr[0])>1:
            h_sample_bits = int(str(arr[1]).strip())
        elif 'BANDS' == arr[0] and len(arr[0])>1: 
            h_bands=int(str(arr[1]).strip())
        if (line.endswith(b'END\r\n') or count>600):
            continuing = 0
    f.close()
    return h_bytes, h_lines,h_line_samples,h_sample_type,h_sample_bits,h_bands


def readImage(file, pixelbytes, sample_type,sample_bits, lines, line_samples, bands):
    f = open(file,'rb')
    filesize = os.fstat(f.fileno()).st_size
    h_bytes = filesize - pixelbytes
    f.seek(h_bytes) # skip past the header bytes
       
    # Assume bands = 1
    print(pixelbytes, lines, line_samples)

    data = f.read(pixelbytes)  # Read raw data bytes
    img = np.frombuffer(data, np.uint16).reshape(lines, line_samples)  # Convert to uint16 NumPy array and reshape to image dimensions.
    img = (img >> 8) + (img << 8)  # Convert from big endian to little endian
      
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)  # Apply demosaicing (convert from Bayer to BGR).

    return img
    

def lin2rgb(im):
    """ Convert im from "Linear sRGB" to sRGB - apply Gamma. """
    # sRGB standard applies gamma = 2.4, Break Point = 0.00304 (and computed Slope = 12.92)    
    # lin2rgb MATLAB functions uses the exact formula [we may approximate it to power of (1/gamma)].
    g = 2.4
    bp = 0.00304
    inv_g = 1/g
    sls = 1 / (g/(bp**(inv_g - 1)) - g*bp + bp)
    fs = g*sls / (bp**(inv_g - 1))
    co = fs*bp**(inv_g) - sls*bp

    srgb = im.copy()
    srgb[im <= bp] = sls * im[im <= bp]
    srgb[im > bp] = np.power(fs*im[im > bp], inv_g) - co
    return srgb


def convert_to_png(full_path):
    # hbytes, hlines, hline_samples, hsample_type, hsample_bits, hbands = readHeader(full_path)
    hbytes = 8
    hlines = 1200
    hline_samples = 1920
    hsample_type = 'blbl'
    hbands = 1
    print("hbytes", hbytes) #taille du header en gros
    print("hlines", hlines) #height
    print("hline_samples", hline_samples)#width
    print("hsample_type", hsample_type) #?balec
    print("hsample_bits", hsample_bits) #16
    print("hbands", hbands) # = 1
    
    numpixels = hlines * hline_samples * hbands
    pixelbytes = numpixels*hsample_bits//8 # // is for integer division
            
    img = readImage(full_path, pixelbytes, hsample_type,hsample_bits, hlines, hline_samples, hbands)

    # Apply gamma correction, and convert to uint8
    img_in_range_0to1 = img.astype(np.float32) / (2**16-1)  # Convert to type float32 in range [0, 1] (before applying gamma correction).
    gamma_img = lin2rgb(img_in_range_0to1)
    gamma_img = np.round(gamma_img * 255).astype(np.uint8)  # Convert from range [0, 1] to uint8 in range [0, 255].

    # cv2.imwrite('gamma_img.png', gamma_img)  # Save image after demosaicing and gamma correction.

    # Show the uint16 image and gamma_img for testing
    # img = img/np.max(img)
    cv2.imshow('img', img)
    # cv2.imshow('gamma_img', gamma_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return img



image_finale = convert_to_png(image_path)