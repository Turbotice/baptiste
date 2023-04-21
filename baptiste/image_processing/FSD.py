# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:32:38 2023

@author: Banquise
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def erodedilate(image, kernel_iteration, kernel_size, save = False, path_images = '', name_save = ''):
    
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    
    img_erosion = cv2.erode(image, kernel, iterations=kernel_iteration)
        
    img_dilatation = cv2.dilate(img_erosion, kernel, iterations=kernel_iteration)
    
    if save :
        plt.imsave(path_images[:-15] + "resultats" + "/" + name_save + "_erodilate.tiff" , img_dilatation, cmap=plt.cm.gray )
        
    return img_dilatation