# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:00:24 2022

@author: Banquise
"""

"""
PROGRAMME QUI DEMODULE ET TROUVE L'AMPLITUDE ET LA LONGUEURE D4ONDE AVEC CA
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
import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.files.file_management as fm
import baptiste.image_processing.image_processing
import baptiste.files.dictionaries as dic
%run parametres_FSD.py

dico = dic.open_dico()

#Va chercher les images et stock leur chemin dasn liste images

path_images, liste_images, titre_exp = fm.import_images(loc,nom_exp, "LAS")

#Importe les paramètres de l'experience

facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = ip.import_param (titre_exp, date)   

mmparpixelx, mmparpixely, mmparpixelz, angle_cam_LAS, mmparpixel = ip.import_calibration(titre_exp,date)          
              
# Creates a file txt with all the parameters of the experience and of the analysis
    
param_complets = ["Paramètres d'adimensionnement :",  "Paramètres d'analyse : ", "mmparpixely = " + str(mmparpixely), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp), "nom_exp = " + str(nom_exp)]


#%%Paramètre de traitement

save = False
display = True

f_exc = round(Tmot)

grossissement = 1 #dico[date][nom_exp]['grossissement']

param_complets.extend([ "grossissement = " + str(grossissement) ,"f_exc = " + str(f_exc) ])

#%%Charge les données (data) et en fait l'histogramme




folder_results = path_images[:-15] + "resultats"
name_file = "positionLAS.npy"
data_originale = np.load(folder_results + "\\" + name_file)

data_originale = np.rot90(data_originale)

debut_las = 400
fin_las = np.shape(data_originale)[0] - 100


t0 = 1
tf = np.shape(data_originale)[1] - 1

if display:
    disp.figurejolie()
    [y,x] = np.histogram((data_originale[debut_las:fin_las,t0:tf]),10000)
    xc= (x[1:]+x[:-1]) / 2
    disp.joliplot("x (pixel)", "Position du laser (pixel)", xc,y, exp = False)
    plt.yscale('log')

# signal contient la detection de la ligne laser. C'est une matrice, avec
# dim1=position (en pixel), dim2=temps (en frame) et valeur=position
# verticale du laser en pixel.

#%% Pre-traitement

savgol = True
im_ref = True

ordre_savgol = 2
taille_savgol = 21
size_medfilt = 51

[nx,nt] = data_originale[debut_las:fin_las,t0:tf].shape


data = data_originale[debut_las:fin_las,t0:tf]


#enlever moyenne pr chaque pixel

if im_ref :
    mean_pixel = np.mean(data,axis = 1) #liste avec moyenne en temps de chaque pixel 
    for i in range (0,nt):
        data[:,i] = data[:,i] - mean_pixel #pour chaque temps, on enleve la moyenne temporelle de chaque pixel


#mise à l'échelle en m
data_m = data *  mmparpixely / 1000

data_m = data_m / grossissement


t = np.arange(0,nt)/facq
x = np.arange(0,nx)*mmparpixelz / 1000

signalsv = np.zeros(data.shape)

#filtre savgol

for i in range(0,nt):  
    signalsv[:,i] = savgol_filter(data_m[:,i], taille_savgol,ordre_savgol, mode = 'nearest')
    if np.mod(i,500)==0:
        print('On processe l image numero: ' + str(i) + ' sur ' + str(nt))
print('Done !')


if savgol :
    data = signalsv
else :
    data = data_m
    
if display:
    plt.figure()
    plt.pcolormesh(data)
    plt.xlabel("Temps (frame)")
    plt.ylabel("X (pixel)")
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (m)')

    
#%% Analyse d'un signal temporel
if True:
    #On prend un point qcq en x, et on regarde le signal au cours du temps.
    figurejolie()
    i = 300
    
    # On va regarder la periode sur signal.
    
    Y1 = fft.fft(data[i,:]- np.mean(data[i,:]))
    
    P2 = abs(Y1/nt)
    P1 = P2[1:int(nt/2+1)]
    P1[2:-1] = 2*P1[2:-1]
    
    
    f = facq * np.arange(0,int(nt/2)) / nt 
    joliplot('f (Hz)', '|P1(f)|', f, P1, title = 'Single-Sided Amplitude Spectrum of X(t)', exp = False)
    
    
#%% Demodulation et amplitude

save = False
# param_complets = param_complets.tolist()
f_exc = 60
amp_demod = []
X = np.linspace(0, nx * mmparpixel/10, nx) #echelle des x en cm

for i in range (nx):
    a = data[i,:]
    amp_demod.append(np.sum(a * np.exp(2*np.pi*1j*f_exc*t))*2/nx)
    
if False : #display : 
    figurejolie()
    joliplot("temps(s)", "Amplitude (m)", t,a, exp = False)

amp_demod = np.asarray(amp_demod)
I = (np.abs(amp_demod))**2 #tableau intensite (avec amp en m)
 
if False : #display :
    figurejolie()
    joliplot(r"x (cm)",r"amplitude (m)", X, np.abs(amp_demod), exp = False, log = False, legend = r"f = " + str(int(f_exc) ) + " Hz")
 
def exppp(x, a, b):
    return a * np.exp(-b * x)

attenuation, pcov = curve_fit (exppp, X, I, p0 = [1,0])
attenuationA = curve_fit (exppp, X, np.abs(amp_demod), p0 = [1,0])

if display :
    figurejolie()
    joliplot(r"x (cm)", r"I", X, I, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
    joliplot(r"x (cm)", r"I", X, exppp(X, attenuation[0][0], attenuation[0][1]), color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation[0][1],4)))
    plt.xscale('linear')
    plt.yscale('log')
    if save :
        plt.savefig(path_images[:-15] + "resultats" + "/" + "I(x)_fitkappa_" + str(round(f_exc)) + "Hz" + ".pdf", dpi = 1)
if save :  
    param_complets.extend(["Resultats attenuation :","f_exc = " + str(f_exc) ,"attenuation = " + str(attenuation[0][1])])

#%% Longueure d'onde

save = False

padding = 12

ampfoispente = np.append( amp_demod * np.exp(attenuationA[0][1] * X), np.zeros(2**padding - nx))
ampfoispente_0 = np.append( (amp_demod) , np.zeros(2**padding - nx))

figurejolie()
joliplot("X (cm)", "Signal", X, np.real(ampfoispente[:nx]),color =2, legend = r'Signal démodulé * atténuation (m)', exp = False)
joliplot("X (cm)", "Signal", X, np.real(ampfoispente_0[:nx]),color = 10, legend = r'Signal démodulé', exp = False)
if save :
    plt.savefig(path_images[:-15] + "resultats" + "/" + "Partie réelle_Signal_démodulé_" + str(round(f_exc)) + "Hz" + ".pdf", dpi = 1)
    


FFT_demod_padding = fft.fft((ampfoispente)-np.mean((ampfoispente)))
FFT_demod_padding_0 = fft.fft((ampfoispente_0)-np.mean((ampfoispente_0)))
FFT_demod = fft.fft((ampfoispente_0[:nx])-np.mean((ampfoispente_0[:nx])))

# P2 = abs(FFT_demod/nt)
# P1 = P2[1:int(nt/2+1)]
# P1[2:-1] = 2*P1[2:-1]
k_padding = np.linspace(0, nx ,2**padding) 
k = np.linspace(0,nx,nx)

figurejolie()
joliplot(r'k ($mm^{-1}$)', r'|P1(k)|',k, np.abs(FFT_demod), exp = False, title = "FFT")

figurejolie()
joliplot(r'k ($mm^{-1}$)', r'|P1(k)|',k_padding, np.abs(FFT_demod_padding),color = 5, exp = False, legend = "FFT zero padding 2**" + str(padding))
joliplot(r'k ($mm^{-1}$)', r'|P1(k)|',k_padding, np.abs(FFT_demod_padding_0),color = 7, exp = False, legend = "FFT zero padding 2**" + str(padding) + " sans atténuation corrigée")
if save :
    plt.savefig(path_images[:-15] + "resultats" + "/" + "FFT_spatiale_0padding_" + str(round(f_exc)) + "Hz" + ".pdf", dpi = 1)
    



#%% Fit second degré
save = True

# Imax = max( np.abs(FFT_demod_padding[2500:]))
# indice1 = np.where( np.abs(FFT_demod_padding) == Imax)

# Imax = max( np.abs(FFT_demod_padding_0))
# indice1 = np.where( np.abs(FFT_demod_padding_0) == Imax)

Imax = max( np.abs(FFT_demod[1000:]))
indice1 = np.where( np.abs(FFT_demod) == Imax)

indice = indice1 [0][0]

y0 = indice #pk + 6 ?

a = 2

z = np.abs(FFT_demod_padding)[y0-a:y0+a]

x = [u for u in range (y0-a,y0+a) - y0 ]

p = np.polyfit(x,z,2)    #on peut utiliser fminsearch pour fitter par une fonction quelconque

imax = -p[1]/(2*p[0])

# longueure_donde = 2**padding * mmparpixel / (indice + imax) / 1000

# longueure_donde = 2**padding * mmparpixel / ((2**padding) - (indice + imax)) / 1000

# longueure_donde = nx * mmparpixel / (indice + imax) / 1000

# longueure_donde = nx * mmparpixel /  (nx - (indice + imax)) / 1000

longueure_donde = nx * mmparpixel / (23.5) / 1000



if save :
    param_complets.extend(["Resultats lambda :","f_exc = " + str(f_exc),"padding = " + str(padding) ,"lambda (m) = " + str(longueure_donde)])


#%% Sauvegarde des parametres et resultats
# param_complets = param_complets.tolist()
if save :
    param_complets.extend(["Paramètres de traitement :",  "debut_las = " + str(debut_las) ,"fin_las = " + str(fin_las),"t0 = " + str(t0) ,"tf = " + str(tf) ])
    param_complets.extend(["savgol = " + str(savgol) ,"im_ref = " + str(im_ref)])
    param_complets.extend(["padding = " + str(padding) ])
    param_complets = np.asarray(param_complets)
    np.savetxt(path_images[:-15] + "resultats" + "/Paramètres_FFT_lambda_kappa" + name_fig_FFT + ".txt", param_complets, "%s")
    

