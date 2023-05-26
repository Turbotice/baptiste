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
%run Functions_FSD.py
%run parametres_FSD.py
%run display_lib_gld.py



dico = open_dico()

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

casse = []
l_onde = []
amplitude = []
exp = []
l_d = []
ldk = []





for date in dico :

    if float(date) > date_min :
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
annotated_ld = True

figurejolie()
for i in range (len(amplitude)) :
    if casse[i] :
        joliplot(r"$\lambda$ (m)", r"Amplitude (m)", l_onde[i], amplitude[i], color = 13)
        if annotated_name :
            plt.annotate (exp[i],(l_onde[i], amplitude[i]))
        if annotated_ld :
            plt.annotate (round(l_d[i], 4),(l_onde[i], amplitude[i]))
    else :
        joliplot(r"$\lambda$ (m)", r"Amplitude (m)", l_onde[i], amplitude[i], color = 14)
        if annotated_name :
            plt.annotate (exp[i],(l_onde[i], amplitude[i]))
        if annotated_ld :
            plt.annotate (round(l_d[i], 4),(l_onde[i], amplitude[i]))

plt.xlim(0,0.45)
plt.ylim(0,0.016)



joliplot(r"$\lambda$ (m)", r"Amplitude (m)", 0, 0, color = 13, legend = 'Casse stationnaire')
joliplot(r"$\lambda$ (m)", r"Amplitude (m)", 0, 0, color = 14, legend = 'Casse pas stationnaire')

x = np.linspace(0,0.42,100)
y = 0.006/0.25 * x 
y_2 = x**2 * 0.10
y_3 = x**3 * 0.42
# y_4 = x ** 1.5 * 0.055

plt.plot(x,y, label = 'Modèle linéaire')
plt.plot(x,y_2, label = 'Modèle quadratique')

plt.plot(x,y_3, label = 'Modèle x3')

# plt.plot(x,y_4, label = 'Modèle x3/2')

plt.legend()
#%% Ajoute points de stage
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
joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées0[:,1], data_traitées0[:,2], color = 16, title = False, legend = r'Casse pas propagatif', exp = True)
joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées1[:,1], data_traitées1[:,2], color = 15, title = False, legend = r'Casse propagatif', exp = True)
# joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées2[:,1], data_traitées2[:,2], color = 2, title = r'Recherche de seuil, $\lambda$(amp)', legend = r'Casse', exp = True)
   
    









