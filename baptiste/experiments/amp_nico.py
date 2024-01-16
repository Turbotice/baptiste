# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:45:09 2023

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


dico = dic.open_dico()

def amp_nico(params, save = True, display = False) :
   
 
    las = panda.read_csv(params['loc_las'] , sep = ',', header= 2)
    
    t_las = np.asarray(las['Protocol TimeStamp'])
    x_las = np.asarray(las['Distance [mm]'])
    
    x_las = x_las - np.mean(x_las)
    t_las = t_las - t_las[0]
    
    if display :
        disp.figurejolie()
        disp.joliplot('t (s)', 'x (mm)', t_las, x_las, exp = False, color = 4) 
        if save :
            plt.savefig(params['path_las'] + 'x_de_t_' + params['file_las'][:-4] + '.pdf')
    
    FFT, f = ft.fft_bapt(x_las, params['facq_las'])
    
    if display :
        disp.figurejolie()
        disp.joliplot('f (Hz)', '|A|', f, np.abs(FFT/len(FFT)), color = 4)
        if save :
            plt.savefig(params['path_las'] + 'fft_' + params['file_las'][:-4] + '.pdf')
    
    #Technique RMS : Marche

    nt = len(t_las)


    [b,a] = sig.butter(3, [params['fexc_1'] - params['d_f_1'], params['fexc_1'] + params['d_f_1']], btype='bandpass', analog=False, output='ba',fs=params['facq_las'])
    params['yfilt_10'] = sig.filtfilt(b, a, x_las)
    
    [b,a] = sig.butter(3, [params['fexc_2'] - params['d_f_2'], params['fexc_2'] + params['d_f_2']], btype='bandpass', analog=False, output='ba',fs=params['facq_las'])
    params['yfilt_20'] = sig.filtfilt(b, a, x_las)
    
    Y1 = fft.fft(x_las)
    Y10 = fft.fft(params['yfilt_10'])
    Y20 = fft.fft(params['yfilt_20'])
    
    P2 = np.abs(Y1)
    P10 = np.abs(Y10)
    P20 = np.abs(Y20)
    
    params['amp_reelle'] = np.sqrt( np.sum ( np.abs(x_las) **2) / nt *2)
    
    params['amp_FFT'] = np.sqrt(np.sum(np.abs(P2 /np.sqrt(nt) )**2 ) / nt*2)
    
    params['amp_FFT_10Hz'] =  np.sqrt( np.sum ( np.abs(params['yfilt_10']) **2) / nt*2)
    
    params['amp_reelle_10Hz'] =  np.sqrt(  np.sum(np.abs(P10 /np.sqrt(nt) )**2 ) / nt*2)
    
    params['amp_FFT_20Hz'] = np.sqrt(  np.sum ( np.abs(params['yfilt_20']) **2) / (nt))
    
    params['amp_reelle_20Hz'] =  np.sqrt(  np.sum(np.abs(P20/np.sqrt(nt) )**2 ) / nt*2)
    
    print('amp_reelle',params['amp_reelle'])
    print('amp_FFT' + str(params['fexc_1']),params['amp_FFT_10Hz'])
    print('amp_FFT' + str(params['fexc_2']),params['amp_FFT_20Hz'])
    
    # print('ratio '+ str(params['fexc_1']) + ' / reel',params['amp_reelle_10Hz'] / params['amp_reelle'])
    # print('ratio '+ str(params['fexc_1']) + ' / ' + str(params['fexc_2']), params['amp_reelle_10Hz'] / params['amp_reelle_20Hz'])
    # print('ratio '+ str(params['fexc_2']) + ' / reel',params['amp_reelle_20Hz'] / params['amp_reelle'])
    
    params['bruit'] = 1 - np.sqrt(params['amp_reelle_10Hz']**2 + params['amp_reelle_20Hz']**2)/params['amp_reelle']
    print('ratio bruit', params['bruit'])
    
    if params['bruit'] > 0.2 :
        print('!!!!  GROS  BRUIT !!!! ')
    
    
    
    
    if save :
        dic.save_dico(params, params['path_las'] + 'params_pointeur_' + params['file_las'][:-4] + '.pkl')
        
    return params
        
    