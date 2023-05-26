# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:53:56 2022

@author: Banquise
"""

import cv2 
import numpy as np
import pandas
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
%run Functions_FSD.py
%run parametres_FSD.py
%run display_lib_gld.py

#%%

exp_1 = np.loadtxt("D:\Banquise\Baptiste\Mesures_autres\d220628\mesure_spray_balance\\Exp_1.txt")
exp_2 = np.loadtxt("D:\Banquise\Baptiste\Mesures_autres\d220628\mesure_spray_balance\\Exp_2.txt")
exp_4 = np.loadtxt("D:\Banquise\Baptiste\Mesures_autres\d220628\mesure_spray_balance\\Exp_4.txt")
exp_5 = np.loadtxt("D:\Banquise\Baptiste\Mesures_autres\d220628\mesure_spray_balance\\Exp_5.txt")

figurejolie()
# joliplot(r'Temps (s)',r'Masse (g)', exp_1[:,0], exp_1[:,1], color = 1, title = False, legend = r'Exp 1', exp = False)
# joliplot(r'Temps (s)',r'Masse (g)', exp_2[:,0], exp_2[:,1], color = 2, title = False, legend = r'Exp 2', exp = False)
# joliplot(r'Temps (s)',r'Masse (g)', exp_4[:,0], exp_4[:,1], color = 3, title = False, legend = r'Exp 4', exp = False)
joliplot(r'Temps (s)',r'Masse (g)', exp_5[:,0], exp_5[:,1], color = 4, title = False, legend = r'Exp 4', exp = True)

p = np.polyfit(exp_5[1:,0], exp_5[1:,1], 1)

x = np.arange(min(exp_5[:,0]), max(exp_5[:,0]))

joliplot(r'Temps (s)',r'Masse (g)', x, p[1] + p[0] * x, color = 2, title = False, legend = r'Fit linéaire', exp = False)
plt.grid()

# pente_evaporation = [0,0,0,0]
pente_evaporation = [-0.0040301875336136985, -0.0018857340122556663, -0.0015411237980185375, -0.0022478039075679484]

#%%

data = np.loadtxt("D:\Banquise\Baptiste\Mesures_autres\d220628\mesure_spray_balance//bonbone_spray_temps.txt")

pbonbonne = data[:,0]
pbalance = data[:,1]
tspray = data[:,2]




def func(x, a):
    return a * x

#masse bonbonne en fonction de masse vernis
figurejolie()
popt, pcov = curve_fit(func, pbalance, pbonbonne)

x = np.arange(0, max(pbalance), 0.01)
joliplot(r'Masse vernis (g)',r'Masse bonbonne (g)', pbalance, pbonbonne, color = 6, title = False, legend = r'Masse vernis (Masse bonbonne)', exp = True)
joliplot(r'Masse vernis (g)',r'Masse bonbonne (g)', x, popt[0] * x, color = 4, title = False, legend = r'Fit linéaire, pente = ' + str(round(popt[0],1)), exp = False)
plt.grid()


#masse bonbonne en fct du temps
figurejolie()
popt, pcov = curve_fit(func, tspray, pbonbonne)

x = np.arange(0, max(tspray), 0.01)

joliplot(r'Temps (s)',r'Masse bonbonne (g)', tspray, pbonbonne, color = 6, title = False, legend = r'Masse bonbonne (temps)', exp = True)
joliplot(r'Temps (s)',r'Masse bonbonne (g)', x, popt[0] * x, color = 4, title = False, legend = r'Fit linéaire, pente = ' + str(round(popt[0],1)), exp = False)

plt.grid()

#%% CM en fct du poids bonbonnne
data = pandas.read_csv("D:\Banquise\Baptiste\Resultats\Coeff_magique\coeff_magique.txt", sep = '\t', header = None)
data = pandas.DataFrame(data).to_numpy()

pds_bonbonne = np.asarray(data[1:,0], dtype = float)
CM = np.asarray(data[1:,2],dtype = float)
x_pds = np.linspace(np.min(pds_bonbonne),np.max(pds_bonbonne), 100)

figurejolie()
joliplot('Poids bonbonne (g)', "Coefficient",pds_bonbonne, CM, legend = 'Exp, std = ' + str(np.std(CM)) + ", mean = " + str(np.mean(CM)), color = 5 )

p = np.polyfit(pds_bonbonne, CM, 1)

joliplot('Poids bonbonne (g)', "Coefficient",x_pds, p[0] * x_pds + p[1], legend = 'Fit linéaire, a = ' + str(p[0]), exp = False, color = 4 )

print(np.std(CM))
print(np.mean(CM))
