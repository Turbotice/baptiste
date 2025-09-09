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
rho = 680


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

rho = 680


blbl, inliers, outliers = fits.fit_ransac(np.log(k), np.log(omega), display = False)
# inliers[25] = True
width = 8.6*4.4/5 #8.6*4/5
disp.figurejolie(width = width)
disp.joliplot(r'k (m$^{-1}$)',r"$\omega$ (m)", data[inliers, 0], data[inliers, 1], log = True, cm = 6, width = 8.6, legend = 'Profilométrie 1D')

# disp.joliplot(r'k (m$^{-1}$)',r"$\omega$ (m)", data[outliers, 0], data[outliers, 1], log = True, color = 15)

k_lin = np.linspace(10, 1000, 100)
omega_cap = rdd.RDD_capilaire(k_lin, tension_surface/rho)


disp.joliplot(r'$k$ (m$^{-1}$)',r"$\omega$ (m)", k_lin, omega_cap, exp = False, log = True, color = 8)

popt = fits.fit(rdd.RDD_flexion, data[inliers, 0], data[inliers, 1], display = False)

disp.joliplot(r'$k$ (m$^{-1}$)',r"$\omega$ (s$^{-1}$)", k_lin, rdd.RDD_flexion(k_lin, popt[0][0]), exp = False, log = True, color = 2)
plt.ylim([20, 3000])
plt.xlim (50 , 800)

# plt.savefig('Y:\Banquise\\Baptiste\\Articles\\Interaction ondes banquise laboratoire\\figures\\Figures_240617\\RDD_v4.svg')

#%% Ajout PIVA6

path_PIVA6 = 'D:\Banquise\\Baptiste\\Resultats_video\\d221104\\d221104_PIVA6_PIV_44sur026_facq151Hz_texp5000us_Tmot010_Vmot410_Hw12cm_tacq020s\\resultats\\lambda_err_fexc221104_PIVA6.txt'


data_PIVA6 = np.loadtxt(path_PIVA6)

lambda_PIVA = data_PIVA6[:,0]
f_PIVA = data_PIVA6[:,2]


disp.joliplot(r'k (m$^{-1}$)',r"$\omega$ (m)",2 * np.pi / lambda_PIVA, 2 * np.pi * f_PIVA, log = True, cm = 3, width = 8.6, legend = 'DIC')

#%% ALL RDD

folder_path = "E:\Baptiste\\Resultats_exp\\All_RDD\\RDD_finales\\"



k = []
omega = []
k_adim = []
omega_adim = []
ld_tot = []
u= 0

disp.figurejolie(width= 8.6*4.4/5)

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        # Charger le fichier (adapté pour colonnes séparées par espaces ou tabulations)
        try:
            data = np.loadtxt(file_path)
        except Exception as e:
            print(f"Impossible de lire {filename} : {e}")
            continue
        
        # Vérification qu'il y a au moins 3 colonnes
        # if data.shape[1] < 3:
        #     print(f"{filename} ne contient pas assez de colonnes.")
        #     continue
        
        # Choix des colonnes
        if filename.startswith("long"):
            pass
            # x = data[:, 0]
            # y = data[:, 1]
        else:
            u+=1
            
            k_exp = 2 * np.pi / data[:, 0]
            omega_exp = 2 * np.pi * data[:, 2]
            
            k = np.append(k, 2 * np.pi /  data[:, 0])
            omega = np.append(omega, 2 * np.pi * data[:, 2])
            
            popt, pcov = curve_fit(rdd.RDD_flexion, k_exp, omega_exp) 
            ld = (popt * 680 / 1000 / g) ** 0.25
            ld_tot = np.append( ld_tot, ld)
            g = 9.81
            k_adim = np.append(k_adim, k_exp * ld )
            omega_adim =np.append(omega_adim, omega_exp / np.sqrt(g / ld) )
            
            disp.joliplot(r'$k L_d$', r'$ \frac{ \omega }{ \sqrt{ g / L_d } }$',  k_exp * ld, omega_exp / np.sqrt(g / ld), cm = int( (ld - 0.002718783732938615) / (0.007465099823376279 - 0.002718783732938615)* 9) , alpha = 0.7, marker_cm= 'x', log = True)
            
disp.joliplot(r'$k L_d$', r'$ \frac{ \omega }{ \sqrt{ g / L_d } }$',  k_lin, k_lin**0.5, color = 8, exp = False, log = True)

disp.joliplot(r'$k L_d$', r'$ \frac{ \omega }{ \sqrt{ g / L_d } }$',  k_lin, k_lin**2.5, color = 2, exp = False, log = True)
plt.xlim(3e-1, 5)
plt.ylim(3e-1,1e2) 
           
disp.figurejolie(width= 8.6)
blbl = np.zeros((2,2))
blbl[0,0] = 0.002718783732938615 * 1000
blbl[1,1] = 0.007465099823376279 * 1000
plt.pcolormesh(blbl, cmap = 'ma_cm')

cbar = plt.colorbar()
cbar.set_label("$L_d$ (mm)")

           

disp.figurejolie(width = 8.6)       
k = np.array(k)
omega = np.array(omega)
disp.joliplot('$k$ (m$^{-1}$)', '$\omega$ (s$^{-1}$)',  k, omega, cm = 5, alpha = 0.5)



disp.figurejolie(width =8.6*4.4/5)
disp.joliplot(r'$k L_d$', r'$ \frac{ \omega }{ \sqrt{ g / L_d } }$',  k_adim, omega_adim, cm = 3, alpha = 0.7, marker_cm= 'x', log = True)
k_lin = np.linspace(np.min(k_adim)*0.9, np.max(k_adim)*1.1, 300)
def f(x, a,b) :
    return np.sqrt( a * x + b * x**5)

poptt, pcovv = curve_fit(f, k_adim, omega_adim)

disp.joliplot(r'$k L_d$', r'$ \frac{ \omega }{ \sqrt{ g / L_d } }$',  k_lin, k_lin**0.5, color = 8, exp = False, log = True)

disp.joliplot(r'$k L_d$', r'$ \frac{ \omega }{ \sqrt{ g / L_d } }$',  k_lin, k_lin**2.5, color = 2, exp = False, log = True)
plt.xlim(3e-1, 5)
plt.ylim(3e-1,1e2) 





