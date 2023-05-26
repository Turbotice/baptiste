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


ttpoints = False


lambdaamp = True
omegaamp = True
Ldpente =False
Ldsurlambdapente = False
Ldomega = False
Ldsurlambdaamp = False
Ldsurlambdacourbure = False
LdlambdaAmp = False
omegalambda = False
hsurlambdapente = True
lambdapente = True
omegapente = False
hpente = False
kpente = True

g = 9.81
tension_surface = 55E-3
rho = 900
Eyoung = 1E7
nu = 0.4



Hfaux = 0.35

#0 ca casse pas, 1 fissure, 2 casse
if ttpoints :
    data = np.loadtxt("D:\Banquise\Baptiste\Resultats\d220617\diagramme_de_phase/diagramme_de_phase_complet_2.txt")
else :
    data = np.loadtxt("D:\Banquise\Baptiste\Resultats\d220617\diagramme_de_phase/diagramme_de_phase_trié.txt")


omega = data[:,0]
lambdaest = data[:,1]
lambdames = data[:,2]
amp = data[:,3]
cassage = data[:,4]
hpese = data[:,5]
hbonbonne = data[:,6]


lambdacomplet = np.zeros(len( lambdames))

data_traitées0 = np.zeros((len(lambdaest), 9)) # casse pas
data_traitées1 = np.zeros((len(lambdaest), 9)) # fissure
data_traitées2 = np.zeros((len(lambdaest), 9)) # casse

#data_traitées avec omega, lambda, Amp, h, Ld, Ld/lambda, pente

for i in range( len(omega)):
    if lambdaest[i] > 0 :
        lambdacomplet[i] = lambdaest[i] / 1000
    if lambdames[i] > 0 :
         lambdacomplet [i] = lambdames[i] / 1000
    
    # if lambdaest[i] < 0 and lambdaest[i] < 0 :
    #     lambdacomplet[i] = 0
     
        
        
    if cassage[i] == 0 :
        data_traitées0[i,0] = omega[i]
        data_traitées0[i,1] = lambdacomplet[i]
        data_traitées0[i,2] = amp[i] / 1000
        
        if hbonbonne[i] > 0 :
            data_traitées0[i,3] = hbonbonne[i] / 1000
        else :
            data_traitées0[i,3] = Hfaux / 1000
            
        data_traitées0[i,4] = ( (Eyoung * data_traitées0[i,3]**3) / (12 * (1 - nu**2) * rho * g) )**(1/4) # Ld
        data_traitées0[i,5] = data_traitées0[i,4] / data_traitées0[i,1] # Ld/lambda
        data_traitées0[i,6] = data_traitées0[i,2] / data_traitées0[i,1] # pente
        data_traitées0[i,7] = data_traitées0[i,6] / data_traitées0[i,1] # courbure
        data_traitées0[i,8] = data_traitées0[i,3] / data_traitées0[i,1] # h/lambda
        
    if cassage[i] == 1 :
        data_traitées1[i,0] = omega[i]
        data_traitées1[i,1] = lambdacomplet[i]
        data_traitées1[i,2] = amp[i] / 1000
        
        if hbonbonne[i] > 0 :
            data_traitées1[i,3] = hbonbonne[i] / 1000
        else :
            data_traitées1[i,3] = Hfaux / 1000
        
        data_traitées1[i,4] = ( (Eyoung * data_traitées1[i,3]**3) / (12 * (1 - nu**2) * rho * g) )**(1/4)
        data_traitées1[i,5] = data_traitées1[i,4] / data_traitées1[i,1] # Ld/lambda
        data_traitées1[i,6] = data_traitées1[i,2] / data_traitées1[i,1] # pente
        data_traitées1[i,7] = data_traitées1[i,6] / data_traitées1[i,1] # courbure
        data_traitées1[i,8] = data_traitées1[i,3] / data_traitées1[i,1] # h/lambda
        
        
        
    if cassage [i] == 2 :
        data_traitées2[i,0] = omega[i]
        data_traitées2[i,1] = lambdacomplet[i]
        data_traitées2[i,2] = amp[i] / 1000
        
        if hbonbonne[i] > 0 :
            data_traitées2[i,3] = hbonbonne[i] / 1000
        else :
            data_traitées2[i,3] = Hfaux / 1000
        
        data_traitées2[i,4] = ( (Eyoung * data_traitées2[i,3]**3) / (12 * (1 - nu**2) * rho * g) )**(1/4) #Ld
        data_traitées2[i,5] = data_traitées2[i,4] / data_traitées2[i,1] # Ld/lambda
        data_traitées2[i,6] = data_traitées2[i,2] / data_traitées2[i,1] # pente
        data_traitées2[i,7] = data_traitées2[i,6] / data_traitées2[i,1] # courbure
        data_traitées2[i,8] = data_traitées2[i,3] / data_traitées2[i,1] # h/lambda
        
        



# Tout est en m !



if lambdaamp :
    figurejolie()
    joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées0[:,1], data_traitées0[:,2], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées1[:,1], data_traitées1[:,2], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées2[:,1], data_traitées2[:,2], color = 4, title = r'Recherche de seuil, $\lambda$(amp)', legend = r'Casse', exp = True)
   
    
 
if Ldpente :
    figurejolie()
    joliplot(r'Pente', r'Ld', data_traitées0[:,6], data_traitées0[:,4], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'Pente', r'Ld', data_traitées1[:,6], data_traitées1[:,4], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'Pente', r'Ld', data_traitées2[:,6], data_traitées2[:,4], color = 4, title = r'Recherche de seuil, Ld (pente)', legend = r'Casse', exp = True)  
    
if omegaamp :
    figurejolie()
    joliplot(r'$\omega$',r'Amp', data_traitées0[:,0], data_traitées0[:,2], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'$\omega$',r'Amp', data_traitées1[:,0], data_traitées1[:,2], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'$\omega$',r'Amp', data_traitées2[:,0], data_traitées2[:,2], color = 4, title = r'Recherche de seuil, $\omega$ (Amp)', legend = r'Casse', exp = True)
    
if Ldomega :
    figurejolie()
    joliplot(r'$\omega$', r'Ld', data_traitées0[:,0], data_traitées0[:,4], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'$\omega$', r'Ld', data_traitées1[:,0], data_traitées1[:,4], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'$\omega$', r'Ld', data_traitées2[:,0], data_traitées2[:,4], color = 4, title = r'Recherche de seuil, Ld ($\omega$)', legend = r'Casse', exp = True)
 
if Ldsurlambdaamp :
    figurejolie()
    joliplot(r'Ld /$\lambda$', r'Amp (m)', data_traitées0[:,5], data_traitées0[:,2], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'Ld /$\lambda$', r'Amp (m)', data_traitées1[:,5], data_traitées1[:,2], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'Ld /$\lambda$', r'Amp (m)', data_traitées2[:,5], data_traitées2[:,2], color = 4, title = r'Recherche de seuil, Ld/$\lambda$ (Amp)', legend = r'Casse', exp = True)

if Ldsurlambdapente :
    figurejolie()
    joliplot(r'Ld /$\lambda$', r'Pente', data_traitées0[:,5], data_traitées0[:,6], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'Ld /$\lambda$', r'Pente', data_traitées1[:,5], data_traitées1[:,6], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'Ld /$\lambda$', r'Pente', data_traitées2[:,5], data_traitées2[:,6], color = 4, title = r'Recherche de seuil, Ld/$\lambda$ (pente)', legend = r'Casse', exp = True)
    
    
if Ldsurlambdacourbure :
    figurejolie()
    joliplot(r'Ld /$\lambda$', r'Courbure', data_traitées0[:,5], data_traitées0[:,7], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'Ld /$\lambda$', r'Courbure', data_traitées1[:,5], data_traitées1[:,7], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'Ld /$\lambda$', r'Courbure', data_traitées2[:,5], data_traitées2[:,7], color = 4, title = r'Recherche de seuil, Ld/$\lambda$ (Courbure)', legend = r'Casse', exp = True)

    
if omegalambda :
    figurejolie()
    joliplot(r'$\omega$', r'$\lambda$', data_traitées0[:,0], data_traitées0[:,1], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'$\omega$', r'$\lambda$', data_traitées1[:,0], data_traitées1[:,1], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'$\omega$', r'$\lambda$', data_traitées2[:,0], data_traitées2[:,1], color = 4, title = r'Recherche de seuil, $\lambda$ ($\omega$)', legend = r'Casse', exp = True)
 
if hsurlambdapente :
    figurejolie()
    joliplot(r'h/$\lambda$', r'Pente', data_traitées0[:,8], data_traitées0[:,6], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'h/$\lambda$', r'Pente', data_traitées1[:,8], data_traitées1[:,6], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'h/$\lambda$', r'Pente', data_traitées2[:,8], data_traitées2[:,6], color = 4, title = r'Recherche de seuil, Pente (h/$\lambda$)', legend = r'Casse', exp = True)

if lambdapente :
    figurejolie()
    joliplot(r'$\lambda$',r'Pente', data_traitées0[:,1], data_traitées0[:,6], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'$\lambda$',r'Pente', data_traitées1[:,1], data_traitées1[:,6], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'$\lambda$',r'Pente', data_traitées2[:,1], data_traitées2[:,6], color = 4, title = r'Recherche de seuil, Pente ($\lambda$)', legend = r'Casse', exp = True)
    #en 1/x

    x = np.linspace(0.02,0.25, 1000)
    unsurx = 0.0030 / x + 0.02*x -0.005
    unsurx = 0.0035 / x #0.0035 est sensé être elongationà la rupture NOPE
    plt.plot(x,unsurx, 'k-')
        
if omegapente :
    figurejolie()
    joliplot(r'$\omega$',r'Pente', data_traitées0[:,0], data_traitées0[:,6], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'$\omega$',r'Pente', data_traitées1[:,0], data_traitées1[:,6], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'$\omega$',r'Pente', data_traitées2[:,0], data_traitées2[:,6], color = 4, title = r'Recherche de seuil, Pente ($\omega$)', legend = r'Casse', exp = True)
 
if hpente :
    figurejolie()
    joliplot(r'h (m)',r'Pente', data_traitées0[:,3], data_traitées0[:,6], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'h (m)',r'Pente', data_traitées1[:,3], data_traitées1[:,6], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'h (m)',r'Pente', data_traitées2[:,3], data_traitées2[:,6], color = 4, title = r'Recherche de seuil, Pente (h)', legend = r'Casse', exp = True)
 
if kpente :
    figurejolie()
    joliplot(r'k (m$^{-1}$)',r'Pente', 1/data_traitées0[:,1] * 2 * np.pi, data_traitées0[:,6], color = 2, title = False, legend = r'Casse pas', exp = True)
    joliplot(r'k (m$^{-1}$)',r'Pente', 1/data_traitées1[:,1] * 2 * np.pi, data_traitées1[:,6], color = 3, title = False, legend = r'Fissure', exp = True)
    joliplot(r'k (m$^{-1}$)',r'Pente', 1/data_traitées2[:,1] * 2 * np.pi, data_traitées2[:,6], color = 4, title = r'Recherche de seuil, Pente(k)', legend = r'Casse', exp = True)
    x = np.linspace(0, 250, 1000)
    unsurx = 0.0005 * x # 0.0005 = e_rupt / 2pi
    plt.plot(x,unsurx, 'k-', label = 'Séparation')
    plt.legend()
 
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
    
    
    
    
    
    
     
    