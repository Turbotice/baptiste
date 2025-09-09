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
# %run Functions_FSD.py
# %run parametres_FSD.py
# %run display_lib_gld.py

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.files.file_management as fm
import baptiste.image_processing.image_processing
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools


date = '220708'
nom_exp = 'IJSP2'
exp = True
exp_type = 'IND'

dico, params, loc = ip.initialisation(date, nom_exp, exp = True, display = False, exp_type = 'IND')

#Va chercher les images et stock leur chemin dasn liste images

path_images, liste_images, titre_exp = ip.import_images(loc,nom_exp,"IND")

#Importe les paramètres de l'experience

facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq= ip.import_param (titre_exp, date)   

mmparpixely, mmparpixelz = ip.import_calibration(titre_exp,date)  

#%% import les data

folder_results = path_images[:-15] + "resultats\\"


data_fz = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220701\d220701_TIPP1_IND_84sur114_facq50Hz_texp5005us_Tmot130_Vmot100_Hw12cm_tacq060s_pb20_pba034/resultats/indentation_contrainte.txt")

data_es = np.loadtxt(folder_results +  "elongation_sigma.txt")
data_imgind = np.loadtxt(folder_results + "image_indentation.txt")
data_imgpds = np.loadtxt(folder_results + "image_poids.txt")

data_las = np.load(folder_results + "positionLAS_IND.npy")
data_las = data_las[300:,:]

[nt,nx] = data_las.shape

# for i in range(nt) :
#     data_las[i, : ] = data_las[i,:] - data_las[0,:]

savgol = True
if savgol :

    ordre_savgol = 2
    taille_savgol = 50
    signalsv = np.zeros(data_las.shape)
    for w in range(0,nt):  
        signalsv[w,:] = savgol_filter(data_las[w,:], taille_savgol,ordre_savgol, mode = 'nearest')
        if np.mod(w,1000)==0:
            print('On processe l image numero: ' + str(w) + ' sur ' + str(nt))
    print('Done !')

    

    data_las = signalsv.copy()
    
#IJSP2
x0_tige = 1126
xf_tige = 1360
x0 = 400
xf = 2050

# #IJSP3
# x0_tige = 930 + 400
# xf_tige = 1200 + 400
# x0 = 940
# xf = 2064

# #IJSP4
# x0_tige = 1300
# xf_tige = 1550
# x0 = 850
# xf = 1800



X = np.linspace(0, nx-1, nx)
T = np.linspace(0, nt-1 , nt)

fill = np.zeros((nt, xf_tige-x0_tige)) * np.nan

data_top = np.append(data_las[:,:x0_tige], fill, axis = 1)
data_top = np.append(data_top, data_las[:,xf_tige:], axis = 1)



data_top = data_top[:,x0:xf]

X = X[x0:xf]

#%% Affichage des données laser 


from scipy.signal import detrend
# disp.figurejolie()
# disp.joliplot('t','x',T, X, table = data_las)


x_milieu = int( x0_tige + (xf_tige-x0_tige)/2 -x0)


# disp.figurejolie()
# for i in range (10,nt,50):
#     disp.joliplot('x','z', X, data_top[i,:], exp = False)
#     plt.pause(0.1)

def longueur_arc(x, y):
    """
    Calcule de la longueur d'arc approximée pour des points discrets (x, y).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Calcul des distances entre points consécutifs-+
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)

    return np.sum(distances)

def f(x,a,b,c):
    return a*x**2 + b * x + c

def f_lin(x,a,b) :
    return a*x + b

len_temps = int(36.9*50)
eps_2 = np.zeros(len_temps) + 1
eps_1 = np.zeros(len_temps) + 1
T = np.linspace(0, len_temps-1 , len_temps) / facq

disp.figurejolie(width = 8.6/ 5 * 3)

for i in [300,900,1500] :#range (0,len_temps): #

    Y = data_top[i,:]
    
    mask = ~np.isnan(X) & ~np.isnan(Y)
    x_clean = X[mask]
    y_clean = detrend(Y[mask])
    
    popt, pcov = curve_fit(f, x_clean, y_clean)
    
    X1 = x_clean[:1126-401]
    X2 = x_clean[1126-400:]

    Y2 =y_clean[1126-400:]
    Y1 =y_clean[:1126-401]
    
    popt1, pcov1 = curve_fit(f_lin, X1, Y1)
    popt2, pcov2 = curve_fit(f_lin, X2, Y2)
    
    y_fit1 = f_lin(X1, popt1[0], popt1[1])
    y_fit2 = f_lin(X2, popt2[0], popt2[1])
    
    
    # print(longueur(0,nx,1000,f))
    
    y_fit = f(X, popt[0], popt[1], popt[2])
    
    if True :
        disp.joliplot(r'X (mm)',r'$\eta$ (mm)', (x_clean - x_clean[0]) * mmparpixely, (y_clean - y_clean[0])* mmparpixelz, cm = 6, marker_cm= 'x', width = 1.5)
        disp.joliplot(r'X (mm)',r'$\eta$ (mm)', (X1 - x0)* mmparpixely, (y_fit1 - y_fit1[0]) * mmparpixelz, exp = False, cm = 3)
        disp.joliplot(r'X (mm)',r'$\eta$ (mm)', (X2 - x0)* mmparpixely, (y_fit2 - y_fit1[0]) * mmparpixelz, exp = False, cm = 3)
    
    if False :
    # disp.joliplot(r'X (mm)',r'$\eta$ (mm)', (X - X[0])* mmparpixely, (y_fit - y_fit[0])* mmparpixelz, exp = False, color = 8)
    
        disp.joliplot(r'X (mm)',r'$\eta$ (mm)', (np.array([x0, X[x_milieu]]) - x0)* mmparpixely, (np.array([y_fit[0], y_fit[x_milieu]]) - y_fit[0]) * mmparpixelz, exp = False, cm = 3)
    
        disp.joliplot(r'X (mm)',r'$\eta$ (mm)', (np.array([X[x_milieu], xf]) - x0)* mmparpixely, (np.array([y_fit[x_milieu], y_fit[-1]]) - y_fit[0])* mmparpixelz, exp = False, cm = 3)
        
        disp.joliplot(r'X (mm)',r'$\eta$ (mm)', (x_clean - x_clean[0]) * mmparpixely, (y_clean - y_clean[0])* mmparpixelz, cm = 6, marker_cm= 'x', width = 1.5)
    
    
    long_1 = longueur_arc([x0, X[x_milieu], xf], [y_fit[0], y_fit[x_milieu],y_fit[-1]])
    # np.sqrt( (x_milieu - x0)**2 + (f(x_milieu, popt[0], popt[1], popt[2]) - f(x0, popt[0], popt[1], popt[2]))**2 ) 
    # + np.sqrt( (xf - x_milieu)**2 + (f(xf, popt[0], popt[1], popt[2]) - f(x_milieu, popt[0], popt[1], popt[2]))**2 )
    
    long_2 = longueur_arc(X,y_fit )

    
    long_plate_1 = np.sqrt( (x_clean[-1] - x_clean[0])**2 + (y_fit[-1] - y_fit[0])**2 )
    long_plate_2 = longueur_arc([x0, xf], [y_fit[0], y_fit[-1]])
    

    eps_2[i] = long_2 / long_plate_1 -1 
    eps_1[i] = long_1 / long_plate_2 -1


disp.figurejolie(width = 8.6/ 5 * 3)
# disp.joliplot('t (s)',r'$\epsilon - 1$', T , eps_2, exp = False, color = 2)
disp.joliplot('t (s)',r'$\epsilon - 1$', T , eps_1, exp = False, cm = 2)
#méthode triangle :


eps_IJSP3 = 0.00121
eps_IJSP2 = 0.00145
eps_IJSP4 = 0.00120 #bof


#%%
TIPP1 = False



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
    disp.figurejolie()
    disp.joliplot( "Indentation (mm)","Contrainte (Pa)", deplacement, contrainte, color = 2, legend = 'Données expérimentales, h = ' + str(round(h_vernis, 3)) + " (mm)", exp = True)

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
    
    for i in range (0,len(imgfor)) :
        indentation.append(-deplacement[int(imgfin[i])])
        
    
    force_adim = force / (h_vernis * (D * rho_vernis * g)**(0.5))
    indent_adim = np.asarray(indentation)/h_vernis
    
    disp.figurejolie()
    
    disp.joliplot( "Indentation (m)","Contrainte (Pa)", indentation, contrainte, color = 2, legend = 'Données expérimentales, h = ' + str(round(h_vernis, 3)) + " (mm)", exp = True)

    disp.figurejolie()
    disp.joliplot( "Indentation/h (m)","Force adim (Pa)", indent_adim, force_adim, color = 2, legend = 'Données expérimentales, h = ' + str(round(h_vernis, 3)) + " (mm)", exp = True)

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

