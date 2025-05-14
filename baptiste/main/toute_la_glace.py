# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:13:01 2024

@author: Banquise
"""

import pandas
import pickle 
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
from scipy.signal import convolve2d
from scipy.signal import savgol_filter, gaussian
from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import stats
import scipy.fft as fft
import os
from PIL import Image
import pandas

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.signal_processing.fft_tools as ft
import baptiste.image_processing.image_processing as imp
import baptiste.math.fits as fits
import baptiste.math.RDD as rdd
import baptiste.files.save as sv
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools



dico = dic.open_dico()

#%%

data = pandas.read_csv('Y:\Banquise\Baptiste\Resultats/toute_la_glace.txt', header = None, sep = '\t')
data = np.asarray(data)

dico_data = {}

u = 0
for key in data[0,:] :
    
    dico_data[key] = np.array(data[1:,u], dtype = float)
    u += 1

rho = 1000
g = 9.81
A = dico_data['A']
lamb = dico_data['lambda']
k = 2 * np.pi / lamb
E = dico_data['E']
h = dico_data['h']
D = E * h**3 / 10
kappa = A * k**2
Ld = (D / rho / g)**0.25
kld = k * Ld
Ak = A * k
kappah = kappa * h

Lk_1 = 2 * np.pi / 4 * Ld
Lk_1 = Lk_1[np.where(kld < 1)]
Lk_2 = lamb / 4
Lk_2 = Lk_1[np.where(kld > 1)]

kappa2Lk1h = kappa**2 * Lk_1 * h
kappa2Lk2h = kappa**2 * Lk_2 * h

Gc_1 = kappa**2 * D * Lk_1 / h
Gc_2 = kappa**2 * D * Lk_2 / h

Gc = np.append(Gc_1, Gc_2)

Gc_1surE = Gc_1 / E[np.where(kld < 1)]
Gc_2surE = Gc_2 / E[np.where(kld > 1)]

GcsurE = np.append(Gc_1surE,Gc_2surE)


critere_GcsurE = np.append([np.where(dico_data['kld'] > 1)], dico_data['h2 kappa2 Lk_1 / 10'][np.where(dico_data['kld'] < 1)])

critere_k2hL = np.append(dico_data['k2Lk2h'][np.where(dico_data['kld'] > 1)], dico_data['k2Lk1h'][np.where(dico_data['kld'] < 1)])

critere_k2hL = np.append(dico_data['k2Lk2h'][np.where(dico_data['kld'] > 1)], dico_data['k2Lk1h'][np.where(dico_data['kld'] < 1)])

disp.figurejolie()
disp.joliplot(r'$\lambda$', r'$\kappa^{2}h^{2}L/10$', dico_data['lambda'], critere_Esurh, log = True, exp = True, color = 5)
plt.ylim(2e-7, 1e-5)

disp.figurejolie()
disp.joliplot(r'$\lambda$', r'$\kappa^{2}h L_k $', dico_data['lambda'], critere_k2hL, log = False, exp = True, color = 5)

#%%CRITERE GC / E

LL = [2.18811E-06,
6.08735E-05,
5.88926E-07,
9.62469E-07,
2.11489E-06]


LAMBDA = [20,
100,
2.5,
6.24,
20]

disp.figurejolie()
disp.joliplot(r'$\lambda$', r'$\kappa^{2}h^{2}L/10$',LAMBDA , LL, log = True, exp = True, color = 5)

#%% CRITERE Gc / D

LLL = [0.000182343,
0.000243494,
0.000196309,
0.000274991,
0.000192263]


disp.figurejolie()
disp.joliplot(r'$\lambda$', r'$\kappa^{2}h L_k $',LAMBDA , LLL, log = True, exp = True, color = 5)



