# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:36:19 2023

@author: Banquise
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
from scipy.signal import convolve2d
from scipy.signal import savgol_filter, gaussian
from scipy.signal import medfilt
from scipy.signal import medfilt2d
from scipy.signal import find_peaks
from scipy.signal import detrend
from scipy.optimize import curve_fit
from scipy import ndimage
from scipy import stats
import scipy.fft as fft
import os
from PIL import Image
from datetime import datetime
from scipy.optimize import minimize
import matplotlib.cm as cm
import pandas as panda
import scipy.io as io
import scipy.signal as sig


import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.files.file_management as fm
import baptiste.image_processing.image_processing
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools
import baptiste.math.RDD as rdd
import baptiste.math.fits as fits
import baptiste.signal_processing.fft_tools as ft
import baptiste.experiments.amp_nico as an

dico = dic.open_dico()


u = 0
for date in ['230725', '230726','230727','230728'] :
    params = {}
    params['date'] = date
    params['loc'] = 'E:\\Nicolas\\d' + date + '\\'
    exp = os.listdir(params['loc'])
    
    for data in exp :
        params['loc_data'] = os.listdir(params['loc'] + data + '\\Data\\')
        
        
        for file_las in params['loc_data'] :
            
            if file_las[-4:] == '.csv' :
                print(data[-6:])
                print(file_las)
                
                params['loc_las'] = params['loc'] + data + '\\Data\\' + file_las
                params['facq_las'] = float(file_las[4:8])
                params['file_las'] = file_las
                params['fexc_1'] = float ( data[data.index("g_") + 2:data.index("g_") + 5]) / 2
                params['fexc_2'] = params['fexc_1'] * 2
                params['d_f_1'] = 2
                params['d_f_2'] = 2
                
                params['path_las'] = params['loc'] + data + '\\Data\\'
                
                params = an.amp_nico(params, save = True, display = True)
                
                u += 1
                print(u)
            
            