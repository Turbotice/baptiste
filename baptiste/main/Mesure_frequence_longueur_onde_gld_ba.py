# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:21:31 2022

@author: Banquise
"""



import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
from scipy.signal import convolve2d
from scipy.signal import medfilt2d
import os
from PIL import Image
from scipy.signal import savgol_filter, gaussian
from scipy.ndimage import gaussian_filter
%run Functions_FSD.py
%run parametres_FSD.py



#Va chercher les images et stock leur chemin dasn liste images

path_images, liste_images, titre_exp = import_images(loc, nom_exp, "LAS")

#Importe les paramètres de l'experience


facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = import_param (titre_exp, date)  

mmparpixelx, mmparpixely, mmparpixelz, angle_cam_LAS, mmparpixel = import_calibration(titre_exp,date)              
              
# Creates a file txt with all the parameters of the experience and of the analysis
    
param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(kernel_size), "kernel_iteration = " + str(kernel_iteration) ,"nbframe = " + str(nbframe) ,"maxsize = " + str(maxsize) , "minsize = " + str(minsize) , "threshold = " + str(threshold) , "sepwb = " + str(sepwb) , "size_crop = " + str(size_crop) , "bordersize = " + str(bordersize), "mmparpixelx = " + str(mmparpixelx), "mmparpixely = " + str(mmparpixely), "mmparpixelz = " + str(mmparpixelz), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp)]

display = False


gradient = False
maximum = True
red_ratio = False
convconv = True

moyenner = False    #faire un moyennage local (convolve2d)
medfilter = True    #mettre un filtre médian et dilate horizontal

save = True
saveplot = False

size_subplot = 3


if nbframe > len(liste_images):
    nbframe = len (liste_images)

#%% 

if display and nbframe > 20 :
    continuer = input ("Attention plus de 20 figures à afficher, continuer ?")
    if not (continuer == "oui") :
        exit()

if convconv :
    maximum = True


im = cv2.imread(path_images + liste_images[0])
im2 = im [:,:,2]
plt.figure()
plt.imshow(im2)


#%% 
# On veut un signal avec, position x, y, et le temps s = f(x,y,t)
# x, la longueur de la nappe laser 
# y la position verticale du laser
# et t un temps correspondant au nombre d'images N/facquisition
[ny,nx] = im2.shape # pour avoir ny le nombre de point verticaux et nx le nombre de points horizontaux
# nt = 200
nt  = len(liste_images) #le nombre d'images
signal = np.zeros((nx,nt))
# Le tableau est initialisé
#%% 
xf = np.linspace(-25,25,50)
fconv = (1-xf**2/sigma)*np.exp(-xf**2/(2*sigma))
fconv = fconv - np.mean(fconv)
kernel = np.ones((1,k_size), np.uint8)

# boucler sur le nombre d'images nt


for i in range(nt): 
    im = cv2.imread(path_images + liste_images[i])
    imred = im[:,:,2]
    if medfilter :
        #filtre médian et dilate en horizontal
        # im_red_traitee = medfilt2d(imred, med_size)
        im_red_traitee = cv2.dilate(im_red_traitee, kernel, iterations=k_iteration)
    for j in range (nx):
        if convconv :
            im_conv = np.convolve(im_red_traitee[:,j], fconv, mode = 'same' )
        if maximum :
            #if convconv
            colonne_k = im_conv
            #trouver le max
            border = 50
            indice =  np.argmax(colonne_k[border:-border])
            indice = indice + border
            #sub_pixellaire
            a=5 #zone pour le fit polynome
            imin = max(1,indice-a)
            imax = min(nx,indice+a)
            z = colonne_k[imin:imax]
            y = [ u for u in range (imin-indice,imax-indice)]
            p = np.polyfit(y,z,2)  #on peut utiliser fminsearch pour fitter par une fonction quelconque
            imax = -p[1]/(2*p[0]) # position y du max
            signal[j,i] = indice + imax
    if np.mod(i,100)==0:
        print('On processe l image numero: ' + str(i) + ' sur ' + str(nt))
print('Done !')

if save :
    np.save(path_images[:-15] + "resultats" + "/Paramètres.txt", param_complets, "%s")
    np.save(path_images[:-15] + "resultats" + "/positionLAS.npy", signal)


