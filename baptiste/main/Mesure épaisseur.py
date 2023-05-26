# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:34:46 2022

@author: Banquise
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
import os
from PIL import Image
from scipy.signal import convolve2d
%run parametres_FSD.py
%run Functions_FSD.py
%run display_lib_gld.py


path_images, liste_images, titre_exp = import_images(loc,nom_exp,"FSD")

#Importe les paramètres de l'experience

facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = import_param (titre_exp)  

mmparpixel = import_calibration(titre_exp,date)           
          
# Creates a file txt with all the parameters of the experience and of the analysis

param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(kernel_size), "kernel_iteration = " + str(kernel_iteration) ,"nbframe = " + str(nbframe) ,"maxsize = " + str(maxsize) , "minsize = " + str(minsize) , "threshold = " + str(threshold) , "sepwb = " + str(sepwb) , "size_crop = " + str(size_crop) , "bordersize = " + str(bordersize), "mmparpixel = " + str(mmparpixel), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp)]



#%%

""" 
Recherche de l'épaisseur globale de la glace
"""
#On prend RDMF par ce qu'on a une ref sans vernis

image_vernis = cv2.imread("D:\Banquise\Baptiste\Resultats_video\d220523\d220523_RDMF1_FSD_38sur55_facq30Hz_texp2112us_Tmot150_Vmot100_Hw12cm_tacq180s\\references\\Image_ref_vernis.tiff",0)
image_ref = cv2.imread("D:\Banquise\Baptiste\Resultats_video\d220523\d220523_RDMF1_FSD_38sur55_facq30Hz_texp2112us_Tmot150_Vmot100_Hw12cm_tacq180s\\references\\Image_ref_sans_vernis.tiff",0)


figurejolie()
joliplot("","",",","",title  = "Image vernis",image = image_vernis)

scale = 255
siz = 21
crop = 1

#idée, séparer en 9 zones


image_ref_crop = image_ref[10: 10 + 1982, 804: 804 + 1212] 

I_min, I_max = np.min(image_ref_crop),np.max(image_ref_crop)
image_ref_crop = ((image_ref_crop-I_min) / (I_max-I_min) * scale)

figurejolie()
joliplot("","",",","",title  = "Image ref crop",image = image_ref_crop)


image_ref_med = image_ref_crop
image_ref_med = image_ref_med[crop : - crop,crop: - crop]

image_ref_med = 255 - image_ref_med 

image_ref_sep = np.zeros(np.shape(image_ref_med))

image_ref_sep [:660,:404] = np.mean(image_ref_crop[:660,:404])
image_ref_sep [660:1320,:404] = np.mean(image_ref_crop[660:1320,:404])
image_ref_sep [1320:,:404] = np.mean(image_ref_crop[1320:,:404])

image_ref_sep [:660,404:808] = np.mean(image_ref_crop[:660,404:808])

image_ref_sep [660:1320,404:808] = np.mean(image_ref_crop[660:1320,404:808])
image_ref_sep [660:1320,404:808] = np.mean(image_ref_crop[660:1320,:])

image_ref_sep [1320:,404:808] = np.mean(image_ref_crop[1320:,404:808])

image_ref_sep [:660,808:] = np.mean(image_ref_crop[:660,808:])
image_ref_sep [660:1320,808:] = np.mean(image_ref_crop[660:1320,808:])
image_ref_sep [1320:,808:] = np.mean(image_ref_crop[1320:,808:])

image_ref_med = 255 - image_ref_sep
image_ref_med = convolve2d (image_ref_med, np.ones((siz,siz))/ siz ** 2 , mode = 'same')


figurejolie()
joliplot("","",",","",title  = "Image ref 9 zones",image = image_ref_sep)

image_vernis = 255 - image_vernis[crop :-crop,crop:- crop]

# I_min, I_max = np.min(image_vernis),np.max(image_vernis)
# image_vernis = ((image_vernis-I_min) / (I_max-I_min) * scale)

figurejolie()
joliplot("","",",","",title  = "Image vernis",image = image_vernis)

figurejolie()
joliplot("","",",","",title  = "Image ref traitée",image = image_ref_med)

# irefsuri0 = (image_vernis) / (image_ref_med)
irefsuri0 = (image_vernis) - (image_ref_med)

figurejolie()
joliplot("","",",","",title  = "Rapport des images",image = irefsuri0)



# I_min, I_max = np.min(im_log),np.max(im_log)
# im_log = ((im_log-I_min) / (I_max-I_min) * scale)
# les points calibrés, x,y,épaisseur (mm)

# calibration = [[655,1555,0.26],[365,148,0.49],[263,950,0.17],[1169,1511,0.48],[263,1696,0.14],[512,1413,0.29]]




Ilog = np.log (irefsuri0)
# I_min, I_max = np.min(Ilog),np.max(Ilog)
# Ilog = ((Ilog-I_min) / (I_max-I_min) * scale)

hmaxsurhmin = np.max(Ilog) / np.min(Ilog)
hmaxsurminquart = np.quantile(Ilog, 0.9) / np.quantile(Ilog, 0.1)

print (hmaxsurhmin)
print (hmaxsurminquart)

figurejolie()
joliplot("","",",","",title  = "Image log",image = Ilog)

Imoy = np.mean(Ilog)
hmoy = 0.106E-3 #en mètres pour RDMF1

alphab = Imoy / hmoy

Hcomplet = Ilog / alphab

plt.pcolormesh(Hcomplet)
plt.colorbar()



