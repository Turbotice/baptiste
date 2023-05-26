# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:38:36 2022

@author: Banquise
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:32:33 2022

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
    
#%%points expérimentaux

lambdaamp = True
Ldpente =False
Ldsurlambdapente = False
Ldomega = False
LdlambdaAmp = False
lambdaamp_plus = False



g = 9.81
tension_surface = 55E-3
rho = 900
Eyoung = 1E7
nu = 0.4


#0 ca casse pas, 1 fissure, 2 casse
data = np.loadtxt("D:\Banquise\Baptiste\Resultats\\220628_diagramme_de_phase\\diagramme_de_phase_trié.txt")

omega = data[:,0]
lambdames = data[:,1]
lambdaest = data[:,2]
amp = data[:,3]
cassage = data[:,4]
hpese = data[:,5]
hbonbonne = data[:,6]


lambdacomplet = np.zeros(len( lambdames))

data_traitées0 = np.zeros((len(lambdaest), 7)) # casse pas
data_traitées1 = np.zeros((len(lambdaest), 7)) # fissure
data_traitées2 = np.zeros((len(lambdaest), 7)) # casse

#data_traitées avec omega, lambda, Amp, h, Ld, Ld/lambda, pente

for i in range( len(omega)):
    if lambdaest[i] == -1 :
        lambdacomplet[i] = lambdames[i] / 1000
    if lambdames[i] == -1 :
        lambdacomplet [i] = lambdaest[i] / 1000
    # if lambdaest[i] ==-1 and lambdaest[i] == -1 :
    #     lambdacomplet[i] = (lambdames[i] + lambdaest[i])/2 / 1000
     
        
        
    if cassage[i] == 0 :
        
        
        if hbonbonne[i] !=-100 : #-1 si on veut mettre h dans les DDP
            data_traitées0[i,3] = hbonbonne[i] / 1000
            data_traitées0[i,0] = omega[i]
            data_traitées0[i,1] = lambdacomplet[i]
            data_traitées0[i,2] = amp[i] / 1000
            data_traitées0[i,4] = ( (Eyoung * data_traitées0[i,3]**3) / (12 * (1 - nu**2) * rho * g) )**(1/4) # Ld
            data_traitées0[i,5] = data_traitées0[i,4] / data_traitées0[i,1] # Ld/lambda
            data_traitées0[i,6] = data_traitées0[i,2] / data_traitées0[i,1] # pente
    
        
        
    if cassage[i] == 1 :
        
        
        if hbonbonne[i] !=-100 :
            data_traitées1[i,3] = hbonbonne[i] / 1000
            data_traitées1[i,0] = omega[i]
            data_traitées1[i,1] = lambdacomplet[i]
            data_traitées1[i,2] = amp[i] / 1000
            data_traitées1[i,4] = ( (Eyoung * data_traitées1[i,3]**3) / (12 * (1 - nu**2) * rho * g) )**(1/4)
            data_traitées1[i,5] = data_traitées1[i,4] / data_traitées1[i,1] # Ld/lambda
            data_traitées1[i,6] = data_traitées1[i,2] / data_traitées1[i,1] # pente

        
        
        
        
    if cassage [i] == 2 :
        
        if hbonbonne[i] !=-100 : 
            data_traitées2[i,3] = hbonbonne[i] / 1000
            data_traitées2[i,0] = omega[i]
            data_traitées2[i,1] = lambdacomplet[i]
            data_traitées2[i,2] = amp[i] / 1000
            data_traitées2[i,4] = ( (Eyoung * data_traitées2[i,3]**3) / (12 * (1 - nu**2) * rho * g) )**(1/4)
            data_traitées2[i,5] = data_traitées2[i,4] / data_traitées2[i,1] # Ld/lambda
            data_traitées2[i,6] = data_traitées2[i,2] / data_traitées2[i,1] # pente
            
        
       
#ajoute les points de MPBV6 et MPPF3-4
# lambda_cassepas = np.append(data_traitées0[:,1], 0.38)
# amp_cassepas = np.append(data_traitées0[:,2], 0.0103)

# lambda_casse = np.append(data_traitées2[:,1], 0.265)
# amp_casse = np.append(data_traitées2[:,2], 0.0068)

# lambda_cassepas = np.append(lambda_cassepas, 0.265)
# amp_cassepas = np.append(amp_cassepas, 0.0063)

 # 1.03 cm casse pas lambda 40 cm
 # 0.63 casse pas
 # 0.68 casse lambda 26.5 cm
# Tout est en m !


if lambdaamp_plus :
    figurejolie()
    joliplot(r'$\lambda$ (m)', r'Amplitude (m)', lambda_cassepas, amp_cassepas, color = 4, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées1[:,1], data_traitées1[:,2], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'$\lambda$ (m)', r'Amplitude (m)', lambda_casse, amp_casse, color = 2, title = r'Recherche de seuil, $\lambda$(amp)', legend = r'Casse', exp = True)
   


if lambdaamp :
    figurejolie()
    joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées0[:,1], data_traitées0[:,2], color = 4, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées1[:,1], data_traitées1[:,2], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées2[:,1], data_traitées2[:,2], color = 2, title = r'Recherche de seuil, $\lambda$(amp)', legend = r'Casse', exp = True)
   
    
if Ldsurlambdapente :
    figurejolie()
    joliplot(r'Pente', r'Ld /$\lambda$', data_traitées0[:,6], data_traitées0[:,5], color = 1, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'Pente', r'Ld /$\lambda$', data_traitées1[:,6], data_traitées1[:,5], color = 2, title = False, legend = r'Fissure', exp = True)
    joliplot(r'Pente', r'Ld /$\lambda$',data_traitées2[:,6], data_traitées2[:,5], color = 3, title = r'Recherche de seuil, Ld/$\lambda$(pente)', legend = r'Casse', exp = True)
    
     
if Ldpente :
    figurejolie()
    joliplot(r'Pente', r'Ld', data_traitées0[:,6], data_traitées0[:,4], color = 1, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'Pente', r'Ld', data_traitées1[:,6], data_traitées1[:,4], color = 2, title = False, legend = r'Fissure', exp = True)
    joliplot(r'Pente', r'Ld', data_traitées2[:,6], data_traitées2[:,4], color = 3, title = r'Recherche de seuil, Ld(pente)', legend = r'Casse', exp = True)  
    
    
if Ldomega :
    figurejolie()
    joliplot(r'$\omega$', r'Ld', data_traitées0[:,0], data_traitées0[:,4], color = 1, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'$\omega$', r'Ld',  data_traitées1[:,0], data_traitées1[:,4], color = 2, title = False, legend = r'Fissure', exp = True)
    joliplot(r'$\omega$', r'Ld',  data_traitées2[:,0], data_traitées2[:,4], color = 3, title = r'Recherche de seuil, Ld($\omega$)', legend = r'Casse', exp = True)
    
if LdlambdaAmp :
    figurejolie()
    
    ax = plt.axes(projection='3d')
    ax.scatter3D(data_traitées0[:,1], data_traitées0[:,2], data_traitées0[:,4], cmap='Greens', label = r'casse pas' )#, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.scatter3D(data_traitées1[:,1], data_traitées1[:,2], data_traitées1[:,4], cmap='Blue', label = r'fissure' )
    ax.scatter3D(data_traitées2[:,1], data_traitées2[:,2], data_traitées2[:,4], cmap='Gray', label = r'casse' )
    ax.set_xlabel(r'$\lambda$ (m)')
    ax.set_ylabel(r'Amp')
    ax.set_zlabel('Ld')
    plt.legend()
    
    
    
    
    
    
     
    