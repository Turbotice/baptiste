# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:14:27 2022

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
import os
from PIL import Image
%run Functions_FSD.py
%run parametres_FSD.py




#%%

#Va chercher les images et stock leur chemin dasn liste images

path_images, liste_images, titre_exp = import_images(loc,nom_exp,"LAS")

#Importe les paramètres de l'experience

facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = import_param (titre_exp)   

mmparpixelx, mmparpixely, mmparpixelz, angle_cam_LAS = import_calibration(titre_exp,date)          
              
# Creates a file txt with all the parameters of the experience and of the analysis
    
param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(k_size_crack), "kernel_iteration = " + str(k_iteration_crack) ,"nbframe = " + str(nbframe) , "minsize_crack = " + str(crack_lenght_min) , "sepwb = " + str(sepwb_cracks) , "size_crop = " + str(size_crop), "mmparpixely = " + str(mmparpixely), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp)]


save_param = True
save_fig = True
save_histo = True

#%%Charge les données et en fait l'histogramme
folder_results = path_images[:-15] + "resultats"
name_file = "positionLAS.npy"
data = np.load(folder_results + "\\" + name_file)
[y,x] = np.histogram((data),10000)
xc= (x[1:]+x[:-1]) / 2
plt.plot(xc,y)
plt.axis([0,800,0,max(y)])

if save_histo :
    plt.savefig(path_images[:-15] + "resultats" + "/" + name_fig_FFT + "histo_LAS.tiff", dpi = 300)

#%% Filtre savgol et median sur toute les images
ordre_savgol = 2
taille_savgol = 101
size_medfilt = 31

[nx,nt] = data.shape

signalsv = np.zeros(data.shape)
signal_medfilt = np.zeros(data.shape)
medfilt_sv = np.zeros(data.shape)
data_mm = data *  mmparpixely


t = np.arange(0,nt)/facq
x = np.arange(0,nx)*mmparpixelz

for i in range(0,nt):
    
    signalsv[:,i] = savgol_filter(data_mm[:,i], taille_savgol,ordre_savgol, mode = 'nearest')
    signal_medfilt[:,i] = medfilt(data_mm[:,i], size_medfilt)
    medfilt_sv[:,i] = savgol_filter(signal_medfilt[:,i], taille_savgol, ordre_savgol, mode = 'nearest')
    if np.mod(i,500)==0:
        print('On processe l image numero: ' + str(i) + ' sur ' + str(nt))
print('Done !')
    
#%%Plot d'une image avec et sans filtre

[nx,nt] = data.shape
x = np.arange(0,nx)*mmparpixelz
fig = plt.figure() 
axes = []
u = 1
size_subplot = 2

i = 300

axes.append( fig.add_subplot(size_subplot, 2, u) )
axes[-1].set_title("signal filtre savegov " + str(i))
plt.axis()
u += 1
plt.plot(x, data_mm[:,i], label = r'signal initial ' + str(i))
plt.plot(x, signalsv[:,i], label = r'signal filtre savegov ' + str(i))
axes[-1].invert_yaxis()
plt.ylabel("mm")
plt.xlabel('mm')

axes.append( fig.add_subplot(size_subplot, 2, u) )
axes[-1].set_title("signal filtre median " + str(i))
plt.axis()
u += 1
# plt.plot(x, data_mm[:,i], label = r'signal initial ' + str(i))
plt.plot(x, signal_medfilt[:,i], label = r'signal filtre median ' + str(i))
axes[-1].invert_yaxis()
plt.ylabel("mm")
plt.xlabel('mm')

axes.append( fig.add_subplot(size_subplot, 2, u) )
axes[-1].set_title("signal filtre median et sv " + str(i))
plt.axis()
u += 1
plt.plot(x, data_mm[:,i], label = r'signal initial ' + str(i))
plt.plot(x, medfilt_sv[:,i], label = r'signal filtre median et sv ' + str(i))
axes[-1].invert_yaxis()
plt.ylabel("mm")
plt.xlabel('mm')

axes.append( fig.add_subplot(size_subplot, 2, u) )
axes[-1].set_title("Image originale "  + str(i))
plt.axis('off')
u += 1
plt.imshow(cv2.imread(path_images + liste_images[i],0), cmap = 'gray')

if save_fig :
    plt.savefig(path_images[:-15] + "resultats/" + name_fig_FFT + "comparaison_filtres_image " + str(i) +".tiff" , dpi = 300)

#%% FFT
[nx,nt] = data.shape
x = np.arange(0,nx)*mmparpixelz
i = 300
fig = plt.figure() 
axes = []
u = 1
size_subplot = 1

s = signal_medfilt[:,i] 

#on soustrait la moyenne de la dérivée
grad = np.gradient(s)
mean_grad = np.mean(grad)
signal_grad = s - mean_grad

#on enlève l'image de base
s = s - signal_medfilt[:,0]



axes.append( fig.add_subplot(size_subplot, 2, u) )
axes[-1].set_title("signal filtre median " + str(i))
plt.axis()
u += 1
plt.plot(x, signal_medfilt[:,i], label = r'signal ' + str(i))
plt.plot(x, s, label = r'signal grad ' + str(i))
axes[-1].invert_yaxis()

#zero_padding
size_padding = pow(2,15)
s = np.append(s,np.zeros(size_padding))
FFT_traitee = abs(np.fft.fft(s))
FFT_base = abs(np.fft.fft(signal_medfilt[:,i] ))
# plt.plot(FFT)

n = FFT_traitee.shape[0]
k_ech = 1 / mmparpixelz
kx = np.linspace(0,k_ech,n)


axes.append( fig.add_subplot(size_subplot, 2, u) )
axes[-1].set_title("FFT " + str(i))
plt.axis()
u += 1
plt.plot(kx,FFT_traitee, label = r'FFT traitée ' + str(i))
# plt.plot(FFT_base, label = r'FFT base ' + str(i))

# derivee
# moyenne derivee
# soustrait

#%% FFT
[nx,nt] = data.shape
x = np.arange(0,nx)*mmparpixelz
i = 300
fig = plt.figure() 
axes = []
u = 1
size_subplot = 1

s = signal_medfilt

#on enlève l'image de base
s = s - signal_medfilt[:,0]


axes.append( fig.add_subplot(size_subplot, 2, u) )
axes[-1].set_title("signal filtre median " + str(i))
plt.axis()
u += 1
plt.plot(x, signal_medfilt[:,i], label = r'signal ' + str(i))
plt.plot(x, s, label = r'signal grad ' + str(i))
axes[-1].invert_yaxis()

#zero_padding
size_padding = pow(2,15)
s = np.append(s,np.zeros(size_padding))
FFT_traitee = abs(np.fft.fft(s))
FFT_base = abs(np.fft.fft(signal_medfilt[:,i] ))
# plt.plot(FFT)

n = FFT_traitee.shape[0]
k_ech = 1 / mmparpixelz
kx = np.linspace(0,k_ech,n)


axes.append( fig.add_subplot(size_subplot, 2, u) )
axes[-1].set_title("FFT " + str(i))
plt.axis()
u += 1
plt.plot(kx,FFT_traitee, label = r'FFT traitée ' + str(i))
# plt.plot(FFT_base, label = r'FFT base ' + str(i))

# derivee
# moyenne derivee
#%%

param_complets.extend(["Paramètres d'affichage et traitement", "save_param : " + str(save_param), "save_fig : " + str(save_fig), "save_histo : " + str(save_histo), "ordre_savgol : " + str(ordre_savgol), "taille_savgol : " + str(taille_savgol), "size_medfilt : " + str(size_medfilt), "numéro_image : " + str(i)])

param_complets = np.array(param_complets)
if save_param :
    np.savetxt(path_images[:-15] + "resultats" + "/Paramètres_" + str(name_fig_FFT) + ".txt", param_complets, "%s")




