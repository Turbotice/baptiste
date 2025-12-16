# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 17:30:18 2025

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
import cv2
from scipy import interpolate
from scipy.signal import savgol_filter

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.signal_processing.fft_tools as ft
import baptiste.image_processing.image_processing as imp
import baptiste.math.fits as fits
import baptiste.math.RDD as rdd
import baptiste.files.save as sv
import baptiste.tools.tools as to


import icewave.tools.matlab2python as m2p
import icewave.baptiste.Fct_drone_1102 as fd

x = np.linspace(0, 6 * np.pi, 100)

disp.figurejolie(width = 8.6 *4 / 5)

for i in range (-4,5,1) :
    eta = i* np.cos(x) / 4 
    
    disp.joliplot(r'$x$', '$\eta$', x, eta, cm = i+4, exp = False)
    
blbl = [[0,1],[0,1]]

disp.figurejolie(width = 8.6 *4 / 5)
plt.pcolormesh(blbl, cmap = 'ma_cm_r')


cbar = plt.colorbar()
cbar.set_ticks([0, 1])
cbar.ax.set_yticklabels(['0', "T"])  