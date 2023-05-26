# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:03:52 2022

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
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import stats
import scipy.fft as fft
import os
from PIL import Image
%run Functions_FSD.py
%run parametres_FSD.py
%run display_lib_gld.py

#%% load data


folder_results = "D:\Banquise\Baptiste\Resultats\d220719"

data_imgind_4 = np.loadtxt(folder_results + "\\" + "image_indentation_IJSP4.txt")
data_imgpds_4 = np.loadtxt(folder_results + "\\" + "image_poids_IJSP4.txt")
data_imgind_2 = np.loadtxt(folder_results + "\\" + "image_indentation_IJSP2.txt")
data_imgpds_2 = np.loadtxt(folder_results + "\\" + "image_poids_IJSP2.txt")
data_imgind_3 = np.loadtxt(folder_results + "\\" + "image_indentation_IJSP3.txt")
data_imgpds_3 = np.loadtxt(folder_results + "\\" + "image_poids_IJSP3.txt")

data_indcontr_TIPP1 = np.loadtxt(folder_results + "\\" + "indentation_contrainte_TIPP1.txt")

data_articletau0 = np.loadtxt(folder_results + '\\' + 'ForceLawTau=0.txt')
data_articletau1 = np.loadtxt(folder_results + '\\' + 'ForceLawTau=1.txt')
data_articletau5 = np.loadtxt(folder_results + '\\' + 'ForceLawTau=5.txt')
data_articletau10 = np.loadtxt(folder_results + '\\' + 'ForceLawTau=10.txt')
data_articletau33 = np.loadtxt(folder_results + '\\' + 'ForceLawTau=33.txt')


#%%
g = 9.81
rho_vernis = 900
D = 3E-6
rho = 1000

E = 150E6
nu = 0.4
tension_surface = 100E-3
surface_vernis = (8.384 / 100) * (11.400 / 100)
poids_vernis = np.asarray([0.34 / 1000,  0.88/1000,  1.14/1000,  0.47/1000]) #h1,h2,h3,h4
h_vernis = poids_vernis/(rho_vernis * surface_vernis) #/0.5# * 1000
surface_tige = (3 / 1000)** 2 * np.pi # diametre = 6mm
D = E * h_vernis**3 / (12 * (1 - nu**2))

tau = tension_surface/ ((rho_vernis * g * D)**(0.5))

#%%
TIPP1 = False
#pour TIPP1  
if TIPP1: 
    force = data_indcontr_TIPP1[:,1] / 1000 * g 
    deplacement = data_indcontr_TIPP1[:,0] / 1000
    contrainte = force / surface_tige
    figurejolie(num_fig = 1)
    joliplot( r"Indentation (m)", r"Contrainte (Pa)", deplacement, contrainte, color = 4, legend = r'Experience 1, h = ' + str(round(h_vernis[0], 6)) + " (m)", exp = True, log = True)
    
    deplacement_adim = deplacement/h_vernis[0]
    force_adim = force / (h_vernis[0] * (D[0] * rho_vernis * g)**(0.5))
    
    figurejolie(num_fig = 2)
    joliplot( r"Indentation/h ", r"Force adim", deplacement_adim, force_adim, color = 4, legend = r'Experience 1, h = ' + str(round(h_vernis[0], 6)) + r" (m), $ \tau $ = " + str(round(tau[0], 3)), exp = True, log = True)



tt = [data_imgpds_2,data_imgpds_3,data_imgpds_4,data_imgind_2,data_imgind_3,data_imgind_4]

for i in range (3) :
    
    #force = (tt[i][:,1] - tt[i][0,1]) / 1000 * g - (rho * g * indentation)
    imgfor = tt[i][:,0]
    
    deplacement = tt[i+3][1] / 1000
    imgdpc = tt[i+3][0]
    
    dif_cam = [23,-20,-20] #IJSP2 23 IJSP3 -20 IJSP4 -20
    sync_img = int(imgdpc[0] - imgfor[0] - dif_cam[i])
    imgfin = imgfor - imgfor[0] - sync_img
        
    contrainte = []
    indentation = []
       
    
        

        
    for j in range (2,len(imgfor)) :
        indentation.append(-deplacement[int(imgfin[j])])
    
    indentation = np.asarray(indentation)
    xx = np.linspace (min(indentation),max(indentation), len(indentation))
    correction = (rho * g * indentation* surface_tige)    
    force = (tt[i][:,1] - tt[i][0,1]) / 1000 * g
    force_adim_sous = ( force[2:]  - correction ) / (h_vernis[i+1] * (D[i+1] * rho_vernis * g)**(0.5))
    force_adim = force[2:]  / (h_vernis[i+1] * (D[i+1] * rho_vernis * g)**(0.5))
    contrainte = force[2:] / surface_tige# / h_vernis[i+1]**(5/2)
    indent_adim = indentation/h_vernis[i+1]
    
    figurejolie(num_fig = 1)   
    joliplot( r"Indentation (m)", r"Contrainte (Pa)", indentation, contrainte, color = i +1, legend = r'Experience ' + str(i + 2) + ", h = " + str(round(h_vernis[i+1], 6)) + " (m)", exp = True, log= True)
    
    figurejolie(num_fig = 2)
    joliplot( r"$\delta$ = $\zeta / h$", r"$F$ = F$/[h(B \rho g)^{1/2}]$", indent_adim, force_adim, color = i +1, legend = r'Experience ' + str(i + 2) + r", h = " + str(round(h_vernis[i+1], 6)) + r" (m), $ \tau $ = " + str(round(tau[i + 1], 3)), exp = True, log = False)
    
    if i == 2:
        plt.plot(data_articletau0[:,0],data_articletau0[:,1], 'k', label = r"Article $ \tau $ = 0")
        plt.plot(data_articletau1[:,0],data_articletau1[:,1], 'k-', label = r'Article $ \tau $ = 1')
        plt.plot(data_articletau5[:,0],data_articletau5[:,1], 'k-', label = r'Article $ \tau $ = 5')
        plt.plot(data_articletau10[:,0],data_articletau10[:,1], 'k-', label = r'Article $ \tau $ = 10')
        plt.plot(data_articletau33[:,0],data_articletau33[:,1], 'k-', label = r'Article $ \tau $ = 33')
    
        if TIPP1:
            plt.xlim (0.1,150)
            plt.ylim(2,15000)
        else :
            plt.xlim (0,20)
            plt.ylim(0,2000)
        plt.title(r'E = ' + str(E)+ r" $\nu$ = " + str(nu))
 
    figurejolie(num_fig = 3)
    joliplot( r"$\delta$ = $\zeta / h$", r"$F$ =( F - $\rho g \zeta ) /[h(B \rho g)^{1/2}]$", indent_adim, force_adim_sous, color = i +1, legend = r'Experience ' + str(i + 2) + r", h = " + str(round(h_vernis[i+1], 6)) + r" (m), $ \tau $ = " + str(round(tau[i + 1], 3)), exp = True, log = False)
    
    if i == 2:
        plt.plot(data_articletau0[:,0],data_articletau0[:,1], 'k', label = r"Article $ \tau $ = 0")
        plt.plot(data_articletau1[:,0],data_articletau1[:,1], 'k-', label = r'Article $ \tau $ = 1')
        plt.plot(data_articletau5[:,0],data_articletau5[:,1], 'k-', label = r'Article $ \tau $ = 5')
        plt.plot(data_articletau10[:,0],data_articletau10[:,1], 'k-', label = r'Article $ \tau $ = 10')
        plt.plot(data_articletau33[:,0],data_articletau33[:,1], 'k-', label = r'Article $ \tau $ = 33')
        
        if TIPP1:
            plt.xlim (0.1,150)
            plt.ylim(2,15000)
        else :
            plt.xlim (0,20)
            plt.ylim(0,2000)
        plt.title(r'E = ' + str(E)+ r" $\nu$ = " + str(nu))
        
    if i == 2:
        
        figurejolie(num_fig = 4)
        joliplot( r"$\zeta$", r"$F$", xx ,force[2:], color = i +1, legend = r'Force Experience ' + str(i + 2) , exp = True, log = False)
        joliplot( r"$\zeta$", r"$Correction$", xx ,correction, color = i +1, legend = r' correction Experience ' + str(i + 2), exp = True, log = False)
        # if TIPP1:
        #     plt.xlim (0.1,150)
        #     plt.ylim(2,15000)
        # else :
        #     plt.xlim (0,20)
        #     plt.ylim(0,2000)
    
    p = np.polyfit(indentation, contrainte, 1)
    figurejolie(num_fig = 5)
    joliplot( r"$\zeta$", r"$F$", xx ,force[2:], color = i +1, legend = r'Force Experience ' + str(i + 2) , exp = True, log = False)
    joliplot( r"$\zeta$", r"$Correction$", xx ,correction, color = i +1, legend = r' correction Experience ' + str(i + 2), exp = True, log = False)
    
   
plt.legend()

#%% fit pente en fct de h






#%% Fit E et nu sur les données de tau = 1


def fct (force, E, nu):
    D = E * h_vernis**3 / (12 * (1 - nu**2))
    return force / (h_vernis[0] * (D * rho_vernis * g)**(0.5))

popt, pcov = curve_fit(fct, data_articletau0[:,0], data_articletau0[:,1])

