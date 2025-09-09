# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:21:10 2025

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
import pandas

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.files.file_management as fm
import baptiste.image_processing.image_processing
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools
import baptiste.math.RDD as rdd
import baptiste.math.fits as fits

dico = dic.open_dico()

path = "E:\Baptiste\\Resultats_exp\\Tableau_params\\Tableau3_Params_231117_240116\\"

file = "tableau_3.txt"

tableau_3 = pandas.read_csv(path + file, header = None, sep = '\t')


tableau_3 = np.asarray(tableau_3)

kappa = np.array(tableau_3[-1,:], dtype = float)
lam = np.array(tableau_3[1,:], dtype = float)
A1 = np.array(tableau_3[2,:], dtype = float)
A2 = np.array(tableau_3[3,:], dtype = float)
a = np.array(tableau_3[4,:], dtype = float)

noms = np.array(tableau_3[0,:])

Acc = np.zeros(len(kappa))
l_ss = np.zeros(len(kappa))
hhh = np.zeros(len(kappa))
Ldd = np.zeros(len(kappa))
lambda_ou = np.zeros(len(kappa))

for i in range (len( noms)) :
    for j in range(len(nom_exps)) :
        if noms[i][:2] in nom_exps[j]:
            Acc[i] = a_s[j]
            l_ss[i] = l_s[j]
            hhh[i] = h[j]
            Ldd[i] = L_d[j]
            lambda_ou[i] = lambda_s[j]
            

plt.figure()

E = 65e6
hh_hh = (Ldd**(4/3) * (10 * 1000 * 9.81)**0.33) / E**0.33

ec = kappa * hh_hh * np.sqrt(lam/12 / np.pi / a / (24 * (1 - 0.4**2)))

plt.plot(lam, ec, 'kx') 


plt.figure()
plt.plot(lam, lambda_ou, 'rx')



















