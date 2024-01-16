# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:39:31 2023

@author: Banquise


Pour plot les paramètes de dictionnaires entre eux
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

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.signal_processing.fft_tools as ft
import baptiste.image_processing.image_processing as imp
import baptiste.math.fits as fits
import baptiste.math.RDD as rdd
import baptiste.files.save as sv



dico = dic.open_dico()

date_min = 230103
date_max = 230120 #AJD ?

var_1 = []
name_1 = 'h'
var_2 = []
name_2 = 'Ld'


for date in dico :
    if date != 'variables_globales' :
        if float(date) >= date_min and float(date) <= date_max :
            print(date)
            for nom_exp in dico[np.str(date)] :
                if name_1 in dico[date][nom_exp] and name_2 in dico[date][nom_exp] :
                    var_1.append(float(dico[date][nom_exp][name_1]))
                    var_2.append(float(dico[date][nom_exp][name_2]))
                    # var_1.append(nom_exp)
                    # var_2.append(dico[date][nom_exp][name_2])


# figurejolie()
# # joliplot(name_1, name_2, var_1, var_2)

# joliplot('', name_2, var_1, var_2)


# casse = np.vstack((var_1, var_2))


#%% Diagramme de phase

dico = dic.open_dico()

date_min = 230103
date_max = 230120 #AJD ?

casse = []
l_onde = []
amplitude = []
exp = []
l_d = []
ldk = []





for date in dico :
    if date.isdigit() : 
        if float(date) > date_min and float(date) < date_max :
            print(date)
    
            for nom_exp in dico[str(date)] :
    
                if 'amp_fracs_fft' in dico[date][nom_exp] and nom_exp != 'MPPF2' and nom_exp != 'MPPF3' :
                    # if "non" in dico[date][nom_exp]['casse'] :
                    print(nom_exp)
                    for j in range (dico[date][nom_exp]['amp_fracs_fft'].shape[0]):
                        if np.shape(dico[date][nom_exp]['amp_fracs_fft'])[1] < 7 :
                            if 'oui' == dico[date][nom_exp]['casse'] :
                                casse.append(True)
                            else :
                                casse.append(False)
                        else :
                            if dico[date][nom_exp]['amp_fracs_fft'][j,6]:
                                casse.append(True)
                            else :
                                casse.append(False)
                            
                       
                        amplitude.append(dico[date][nom_exp]['amp_fracs_fft'][j,4] * np.sqrt(2))# * np.pi /dico[date][nom_exp]['lambda'] )
                        exp.append(nom_exp)
                        l_d.append(dico[date][nom_exp]['Ld'])
                        ldk.append(2 * np.pi * dico[date][nom_exp]['Ld'] /dico[date][nom_exp]['lambda'])
                        l_onde.append(dico[date][nom_exp]['lambda'])
  
    
#%% Graph
annotated_name = False
annotated_ld = False

# figurejolie()
for i in range (len(amplitude)) :
    if casse[i] :
        disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", l_onde[i], amplitude[i], color = 13)
        if annotated_name :
            plt.annotate (exp[i],(l_onde[i], amplitude[i]))
        if annotated_ld :
            plt.annotate (round(l_d[i], 4),(l_onde[i], amplitude[i]))
    else :
        disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", l_onde[i], amplitude[i], color = 14)
        if annotated_name :
            plt.annotate (exp[i],(l_onde[i], amplitude[i]))
        if annotated_ld :
            plt.annotate (round(l_d[i], 4),(l_onde[i], amplitude[i]))

plt.xlim(0,0.45)
plt.ylim(0,0.016)



disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", 0, 0, color = 13, legend = 'Casse stationnaire')
disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", 0, 0, color = 14, legend = 'Casse pas stationnaire')

# x = np.linspace(0,0.42,100)
# y = 0.006/0.25 * x 
# y_2 = x**(1.5) * 0.056
# y_3 = x**2 * 0.1
# # y_4 = x ** 1.5 * 0.055

# plt.plot(x,y, label = 'Modèle linéaire')
# plt.plot(x,y_2, label = 'Modèle 1.5')

# plt.plot(x,y_3, label = 'Modèle 2')

# # plt.plot(x,y_4, label = 'Modèle x3/2')

# plt.legend()
#%% Ajoute points de stage
data = np.loadtxt("D:\Banquise\Baptiste\Resultats\\220628_diagramme_de_phase\\diagramme_de_phase_trié.txt")

omega = data[:,0]
lambdames = data[:,1]
lambdaest = data[:,2]
amp = data[:,3]
cassage = data[:,4]
hpese = data[:,5]
hbonbonne = data[:,6]

disp.figurejolie()

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
            # data_traitées0[i,4] = ( (Eyoung * data_traitées0[i,3]**3) / (12 * (1 - nu**2) * rho * g) )**(1/4) # Ld
            data_traitées0[i,5] = data_traitées0[i,4] / data_traitées0[i,1] # Ld/lambda
            data_traitées0[i,6] = data_traitées0[i,2] / data_traitées0[i,1] # pente
    
        
        
    if cassage[i] == 1 or cassage [i] == 2 :
        
        
        if hbonbonne[i] !=-100 :
            data_traitées1[i,3] = hbonbonne[i] / 1000
            data_traitées1[i,0] = omega[i]
            data_traitées1[i,1] = lambdacomplet[i]
            data_traitées1[i,2] = amp[i] / 1000
            # data_traitées1[i,4] = ( (Eyoung * data_traitées1[i,3]**3) / (12 * (1 - nu**2) * rho * g) )**(1/4)
            data_traitées1[i,5] = data_traitées1[i,4] / data_traitées1[i,1] # Ld/lambda
            data_traitées1[i,6] = data_traitées1[i,2] / data_traitées1[i,1] # pente

        
        
        
        
    # if cassage [i] == 2 :
        
    #     if hbonbonne[i] !=-100 : 
    #         data_traitées2[i,3] = hbonbonne[i] / 1000
    #         data_traitées2[i,0] = omega[i]
    #         data_traitées2[i,1] = lambdacomplet[i]
    #         data_traitées2[i,2] = amp[i] / 1000
    #         data_traitées2[i,4] = ( (Eyoung * data_traitées2[i,3]**3) / (12 * (1 - nu**2) * rho * g) )**(1/4)
    #         data_traitées2[i,5] = data_traitées2[i,4] / data_traitées2[i,1] # Ld/lambda
    #         data_traitées2[i,6] = data_traitées2[i,2] / data_traitées2[i,1] # pente
            
        
       
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



# figurejolie()
disp.joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées0[:,1], data_traitées0[:,2], color = 16, title = False, legend = r'Intact', exp = True)
disp.joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées1[:,1], data_traitées1[:,2], color = 15, title = False, legend = r'Fracturé', exp = True)
# joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées2[:,1], data_traitées2[:,2], color = 2, title = r'Recherche de seuil, $\lambda$(amp)', legend = r'Casse', exp = True)
   
    
#%% Plus joli

disp.figurejolie()
#Données de stage
disp.joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées0[:,1], data_traitées0[:,2], color = 16, title = False, legend = r'Intact', exp = True)
disp.joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées1[:,1], data_traitées1[:,2], color = 15, title = False, legend = r'Fracturé', exp = True)

#Données stationnaire

for i in range (len(amplitude)) :
    if casse[i] :
        disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", l_onde[i], amplitude[i], color = 15)
        if annotated_name :
            plt.annotate (exp[i],(l_onde[i], amplitude[i]))
        if annotated_ld :
            plt.annotate (round(l_d[i], 4),(l_onde[i], amplitude[i]))
    else :
        disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", l_onde[i], amplitude[i], color = 16)
        if annotated_name :
            plt.annotate (exp[i],(l_onde[i], amplitude[i]))
        if annotated_ld :
            plt.annotate (round(l_d[i], 4),(l_onde[i], amplitude[i]))

plt.xlim(0,0.45)
plt.ylim(0,0.016)


#FIT
x = np.linspace(0,0.42,100)
y = 0.006/0.25 * x**(0.75) * 0.6
y_2 = x**(1.5) * 0.056
y_3 = x**2 * 0.1
# y_4 = x ** 1.5 * 0.055

# joliplot(r"$\lambda$ (m)", r"Amplitude (m)", x,y, color = 3, legend = r'Modèle contrainte visqueuse ($\frac{3}{4}$)', exp = False)
# joliplot(r"$\lambda$ (m)", r"Amplitude (m)",x,y_2,color = 8, legend = r'Modèle flexion ($\frac{3}{2}$)', exp = False)
# joliplot(r"$\lambda$ (m)", r"Amplitude (m)",x,y_3,color = 6, legend = r'Modèle $^{2}$', exp = False)

# plt.plot(x,y_4, label = 'Modèle x3/2')
plt.grid()
plt.legend()
#%%

disp.figurejolie()

k = np.logspace(-7, 3)

def RDD_full(k,drhoh, Dsurrho, Tsurrho, H, g):
    return np.sqrt( np.tanh(H * k) * ((g * k + Tsurrho * k**3 + Dsurrho * k **5) * (( 1 + drhoh * k)**(-1))) )

disp.joliplot(r'$\omega$', r'k',k, RDD_full(k,0.001,1,10000,100, 10), exp = False, log = True, color = 8)

plt.axis('equal')

#%%

disp.figurejolie()

k = np.logspace(-8, 3)

def RDD_full(k, Dsurrho, Tsurrho, g):
    return np.sqrt( (g * k + Tsurrho * k**3 + Dsurrho * k **5) )

disp.joliplot(r'$\omega$', r'k',k, RDD_full(k,1e6,10000000,10), exp = False, color = 8, log = True)

# plt.axis('equal')

#%% RDD fictive FULLLL

x1 = np.logspace(-1,0,100)
x2 = np.logspace(0,1,100)
x3 = np.logspace(1,2,100)
x4 = np.logspace(2,3,100)
x5 = np.logspace(3,4,100)

disp.figurejolie()

disp.joliplot( r'k $(m^{-1})$',r'$\omega$ (Hz)',x1, x1, exp = False, color = 8, log = True)
disp.joliplot( r'k $(m^{-1})$',r'$\omega$ (Hz)',x2, x2**(0.5), exp = False, color = 8, log = True)
disp.joliplot( r'k $(m^{-1})$',r'$\omega$ (Hz)',x3, x3**(1.5)/10, exp = False, color = 8, log = True)
disp.joliplot( r'k $(m^{-1})$',r'$\omega$ (Hz)',x4, x4**(2.5)/1000, exp = False, color = 8, log = True)
disp.joliplot( r'k $(m^{-1})$',r'$\omega$ (Hz)',x5, x5**2/np.sqrt(1000), exp = False, color = 8, log = True)

plt.grid()

# plt.axis('equal')

#%% RDD fictive g et D

x1 = np.logspace(-1,0,100)
x2 = np.logspace(0,1,100)

disp.figurejolie()

disp.joliplot(r'k $(m^{-1})$',r'$\omega$ (Hz)',x1, x1**(0.5), exp = False, color = 8, log = True)
disp.joliplot(r'k $(m^{-1})$',r'$\omega$ (Hz)',x2, x2**(2.5), exp = False, color = 8, log = True)

# plt.axis('equal')

#%% Coeff magique

disp.figurejolie()
data_coeff = np.loadtxt("D:\Banquise\Baptiste\\Resultats\\Coeff_magique\\coeff_magique.txt", skiprows= 1)

poids_bonbonne = data_coeff[:,0] - data_coeff[:,1]

poids_déposé = poids_bonbonne /data_coeff[:,2] 


disp.joliplot(r'Masse perdue (g)',r'Masse déposée (g)',   poids_déposé, poids_bonbonne, color = 2)

def lin (x,a) :
    return a * x 

popt, pcov = fits.fit(lin, poids_déposé, poids_bonbonne,  display = True, err = False, nb_param = 1, p0 = [0], bounds = False, 
        zero = True, th_params = False, xlabel = r'Masse déposée (g)', ylabel = r'Masse perdue (g)', legend_data = r'Experimental Data', legend_fit = 'a = ')





























