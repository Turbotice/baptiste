# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 18:10:46 2022

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

#Va chercher les images et stock leur chemin dasn liste images

path_images, liste_images, titre_exp = import_images(loc,nom_exp,"IND")

#Importe les paramètres de l'experience

facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = import_param (titre_exp)   

mmparpixely, mmparpixelz = import_calibration(titre_exp,date)  
        
#%% import les data

folder_results = path_images[:-15] + "resultats"


# data_fz = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220701\d220701_TIPP1_IND_84sur114_facq50Hz_texp5005us_Tmot130_Vmot100_Hw12cm_tacq060s_pb20_pba034/resultats/indentation_contrainte.txt")

data_es = np.loadtxt(folder_results + "\\" + "elongation_sigma.txt")
data_imgind = np.loadtxt(folder_results + "\\" + "image_indentation.txt")
data_imgpds = np.loadtxt(folder_results + "\\" + "image_poids.txt")


#%%
TIPP1 = True


g = 9.81
rho_vernis = 900
D = 3E-6
nu = 0.4
surface_vernis = (8.384 / 100) * (11.400 / 100)

poids_vernis = 0.34 / 1000 # 0.88 IJSP2, 1.14 IJSP3, 0.47 IJSP4, 0.34 TIPP1

h_vernis = poids_vernis/(rho_vernis * surface_vernis) * 1000
surface_tige = (3 / 1000)** 2 * np.pi # diametre = 6mm

if TIPP1 :
    poids_vernis = 0.34 / 1000
    h_vernis = poids_vernis/(rho_vernis * surface_vernis) * 1000
    elongation = data_es[0]
    rayon = data_es[1]
    force = data_fz[:,1] / 1000 * g #pour TIPP1
    deplacement = data_fz[:,0]
    contrainte = force / surface_tige
    figurejolie()
    joliplot( "Indentation (mm)","Contrainte (Pa)", deplacement, contrainte, color = 2, legend = 'Données expérimentales, h = ' + str(round(h_vernis, 3)) + " (mm)", exp = True)

else :
    force = (data_imgpds[:,1] - data_imgpds[0,1]) / 1000 * g
    imgfor = data_imgpds[:,0]
    
    deplacement = data_imgind[1] / 1000
    imgdpc = data_imgind[0]
    
    dif_cam = -20 #IJSP2 23 IJSP3 -20 IJSP4 -20
    sync_img = int(imgdpc[0] - imgfor[0] - dif_cam)

    imgfin = imgfor - imgfor[0] - sync_img

    deplacement_adim = deplacement/h_vernis
    
    contrainte = []
    indentation = []
    
    contrainte = force[2:] / surface_tige
    
    
    """contrainte ind"""
    
    for i in range (2,len(imgfor)) :
        indentation.append(-deplacement[int(imgfin[i])])
        
        
    force_adim = force / (h_vernis * (D * rho_vernis * g)**(0.5))
    indent_adim = indentation/h_vernis
    
    figurejolie()
    
    joliplot( "Indentation (m)","Contrainte (Pa)", indentation, contrainte, color = 2, legend = 'Données expérimentales, h = ' + str(round(h_vernis, 3)) + " (mm)", exp = True)

    figurejolie()
    joliplot( "Indentation/h (m)","Force adim (Pa)", indent_adim, force_adim, color = 2, legend = 'Données expérimentales, h = ' + str(round(h_vernis, 3)) + " (mm)", exp = True)

#%% save data




#%%
"""depalcement force adim"""

# figurejolie()
# joliplot("Deplacement/h ","Pression (Pa)",  deplacement_adim, force_adim, color = 1, legend = 'courbe', exp = True, log = True)
    
# p = np.polyfit(deplacement_adim,force_adim ,1)  

# x= np.arange(min(deplacement_adim),max(deplacement_adim), 0.00001)  

# joliplot( "Deplacement (mm)","Pression (Pa)", x, p[0] * x, color = 2, legend = 'fit', exp = False)

"""deplpacement contrainte"""
# figurejolie()

# joliplot("Deplacement/h ","Pression (Pa)",  deplacement_adim, contrainte, color = 1, legend = 'courbe', exp = True, log = True)


# p = np.polyfit(deplacement_adim,contrainte ,1)  

# x= np.arange(min(deplacement_adim),max(deplacement_adim), 0.00001)  
# joliplot( "Deplacement (mm)","Pression (Pa)", x, p[0] * x, color = 2, legend = 'fit', exp = False)

"""elongation contrainte"""
figurejolie()
joliplot( "Elongation","Contrainte (Pa)", elongation[2:], contrainte[2:], color = 1, legend = 'exp', exp = True)

p = np.polyfit(elongation[2:-25],contrainte[2:-25] ,1)  

x= np.arange(min(elongation[2:]),max(elongation[2:]), 0.00001) 
 
joliplot( "Elongation","Contrainte (Pa)", x, p[0] * x + p[1], color = 2, legend = 'fit, E = '+ str(round(p[0],0)), exp = False)
D = p[0] * h_vernis**3 / (12 * (1 - nu**2))
plt.grid()

