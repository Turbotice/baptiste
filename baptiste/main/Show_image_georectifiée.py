# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 15:28:36 2025

@author: Banquise
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from scipy.signal import find_peaks
from skimage.measure import profile_line
import scipy.fft as fft
import scipy.io as io
import h5py
import cv2 as cv
from scipy import interpolate
from scipy.signal import savgol_filter

import os
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
from datetime import datetime, time , timedelta
import pytz
import glob 
import imageio as iio
import cv2 as cv
import h5py
from scipy.io import loadmat

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.signal_processing.fft_tools as ft
import baptiste.image_processing.image_processing as imp
import baptiste.math.fits as fits
import baptiste.math.RDD as rdd
import baptiste.files.save as sv


import icewave.tools.matlab2python as m2p
import icewave.baptiste.Fct_drone_1102 as fd
import icewave.tools.matlab2python as mat2py
import icewave.drone.drone_projection as dp 

import pickle

# filelist_tif = glob.glob(base + '*.tiff')
base = 'Y:/Banquise/Sebastien/Calibration_intrinseque/Calib_oblique_bernache_2/'
# filelist_tif = glob.glob(base + '*.tif')
filelist_param = glob.glob(base + 'Param*.mat')

base = 'Y:\Banquise\\BicWin2024\\PIV_images\\0211\\mesange\\'

def plot_img(X, Y, t, renvoyer = False) : 

    i0 = t + 520
    i0 = str(11496+ i0)
    
    frame = base + 'im_' + i0 + '.tiff'
    
    # frame = filelist_tif[i0]
    
    
    img = cv.imread(frame)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    
    focale = 2700
    [ny,nx,nc] = np.shape(img) 
    
    x_edges = np.arange(0,nx + 1)
    y_edges = np.arange(0,ny + 1)
    
    x0 = (nx + 1) / 2
    y0 = (ny + 1) / 2
    
    Yedges,Xedges = np.meshgrid(y_edges,x_edges,indexing = 'ij')
    
    Yedges = Yedges[:-1,:-1]
    Xedges = Xedges[:-1,:-1]
    
    # disp.figurejolie()
    # plt.imshow(img)
    
    param_file = filelist_param[0]
    
    with h5py.File(param_file, 'r') as fmat:
        param_dict = {}
        
        print('Top-level keys : ', list(fmat.keys()))
    
        param_dict = mat2py.mat_to_dict(fmat['param'],fmat['param'])
    
    # compute real coordinates for each pixels of the image 
    
    param_dict['H'] = 90.300
    param_dict['alpha_0'] = 1.044 * 180 / np.pi
    
    Xreal,Yreal = dp.projection_real_space(Xedges,Yedges,x0,y0,param_dict['H'],param_dict['alpha_0']*np.pi/180,focale)
    
    # # compute object position
    # POS_realx,POS_realy = dp.projection_real_space(POS[:,1],POS[:,2],x0,y0,
    #                                                param_dict['H'],param_dict['alpha_0']*np.pi/180,focale)
    
    disp.figurejolie(width = 8.6 * 104.8 / 120 * 4, height = 8.6 * 73.6 / 120 * 4)
    # Affichage
    plt.pcolormesh(Xreal,Yreal, img[:,:,1], cmap = 'Greys_r')
    # norm = Normalize(0,1)
    # plt.pcolormesh(Xreal, Yreal, img[:,:,0], cmap='Greys_r', norm=norm, shading='auto', alpha=0.3)
    # # canal vert
    # plt.pcolormesh(Xreal, Yreal, img[:,:,1], cmap='Greys_r', norm=norm, shading='auto', alpha=0.3)
    # # canal bleu
    # plt.pcolormesh(Xreal, Yreal, img[:,:,2], cmap='Greys_r', norm=norm, shading='auto', alpha=0.3)
    
    # img = (img - img.min()) / (img.max() - img.min())
    # img = np.transpose(img, (1, 0, 2))
    # plt.pcolormesh(Xreal, Yreal, np.zeros_like(Xreal), facecolors=img, shading='auto')

    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    
    plt.show()
    
    if renvoyer :
        return Xreal, Yreal, img