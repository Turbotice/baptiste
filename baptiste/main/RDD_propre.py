# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:07:00 2023

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


import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.files.file_management as fm
import baptiste.image_processing.image_processing
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools
import baptiste.math.RDD as rdd
import baptiste.math.fits as fits

dico = dic.open_dico()

#%%

date = '221024'

exp_à_traiter = ["DAP07","DAP08","DAP09","DAP10"] #[ "DAP10","DAP09","DAP07","PIVA6"]#["DAP01","DAP02","DAP03","DAP04","DAP05","DAP06","DAP07","DAP08","DAP09","DAP10"] #["all","DAP"] 
exp_séparées = False
ATT = False
RDD = True

loc = 'D:\Banquise\Baptiste\Resultats_video\\d' + date + '\\'

g = 9.81
tension_surface = 50E-3
rho = 900


m = (226.93-188.24)/3.1

S = 0.4*.26
h = m / (S * rho) #importer h depuis les paramètres

liste_fichiers = np.zeros(len(exp_à_traiter), dtype = object)

for i in range (len(exp_à_traiter)) :
    u = os.listdir(loc)
    for exp in u :
        if exp_à_traiter[i] in exp :
            path_resultats = loc + exp + '\\resultats'
            liste_fichiers[i] = np.loadtxt(path_resultats + '\\' + "lambda_err_fexc221024_" + exp_à_traiter[i] + '.txt')
            

#%%

long_onde = np.array([])
f = np.array([])
err_lambda = np.array([])

for i in liste_fichiers :
    long_onde = np.append(long_onde, i[:,0])
    err_lambda = np.append(err_lambda, i[:,1])
    f = np.append(f,  i[:,2])
    
omega = 2 * np.pi * f
k = 2 * np.pi / long_onde
err_k = 0


#%%
data = np.stack((k,omega), axis = -1)  




blbl, inliers, outliers = fits.fit_ransac(np.log(k), np.log(omega), display = False)
# inliers[25] = True
width = 6.3 #8.6*4/5
disp.figurejolie(width = width)
disp.joliplot(r'k (m$^{-1}$)',r"$\omega$ (m)", data[inliers, 0], data[inliers, 1], log = True, color = 14, width = width)

# disp.joliplot(r'k (m$^{-1}$)',r"$\omega$ (m)", data[outliers, 0], data[outliers, 1], log = True, color = 15)

k_lin = np.linspace(10, 1000, 100)
omega_cap = rdd.RDD_capilaire(k_lin, tension_surface/rho)


disp.joliplot(r'$k$ (m$^{-1}$)',r"$\omega$ (m)", k_lin, omega_cap, exp = False, log = True, color = 8)

popt = fits.fit(rdd.RDD_flexion, data[inliers, 0], data[inliers, 1], display = False)

disp.joliplot(r'$k$ (m$^{-1}$)',r"$\omega$ (s$^{-1}$)", k_lin, rdd.RDD_flexion(k_lin, popt[0][0]), exp = False, log = True, color = 2)
plt.ylim([10, 5000])

plt.savefig('Y:\Banquise\\Baptiste\\Articles\\Interaction ondes banquise laboratoire\\figures\\Figures_240617\\RDD_v4.svg')






