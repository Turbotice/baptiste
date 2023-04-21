# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:31:10 2022

@author: Turbots
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import filters
import os
from scipy.optimize import curve_fit
from scipy import stats
from PIL import Image
import pickle
from skimage.morphology import skeletonize


'''
Fonctions de démarrage
'''

def import_images(loc, nom_exp, exp_type, nom_fich = "\image_sequence\\"):
    fichiers = []                             
    liste_images = []

    fichiers = os.listdir(loc)        

    for j in range (len (fichiers)):
        if nom_exp == fichiers[j][8:13] :
            if exp_type in fichiers[j]:
                titre_exp = fichiers[j]
                path_images = str(loc + fichiers [j] + nom_fich)

    liste_images = os.listdir(path_images)

    print (path_images)
      
    #Créé un répertoire pour les résultats
         
    # if os.path.isdir(path_images[:-15] + "resultats") == False:
    #       os.mkdir(path_images[:-15] + "resultats")
          
    return path_images, liste_images,titre_exp


    


'''
Fonctions pour FFT
'''

def find_best_lin (Y, X = False, range_taille = 3, pas_taille = 12):
    #prend une courbe et trouve la meilleure zone avec un fit linéaire, zone de taille 1 / 1 + range_taille et on decale de 1/pas_taille à chaque fois
     
    R = []
    length = len (Y)
    
    for taille_test in range (1 , range_taille + 1) :
        for pos_test in range (pas_taille) :
            if 1 >= (1/taille_test + pos_test/pas_taille):
                 
                zone_test = Y [int(length/ pas_taille * pos_test) : int(length *  (1/taille_test + pos_test/pas_taille) ) ]
                if np.mean(X) == False :
                    x = np.arange(len (zone_test))
                else :
                    x = X [int(length/ pas_taille * pos_test) : int(length *  (1/taille_test + pos_test/pas_taille) ) ]
                
                slope, intercept, r_value, p_value, std_error = stats.linregress(x,zone_test)
                r = r_value ** 2
                R.append ([[slope,intercept], abs(r), taille_test, pos_test])
                
    R = np.asarray(R)    
    max_r = max(R[:,1]) #coeff de corélation le meilleur
    max_corel = R[np.argmax(R[:,1])]   
    p = max_corel[0] #[pente, origine] où il y a le meilleur coeff correl
    
    best_taille = max_corel[2] #taille de la zone la meilleure
    best_pos = max_corel[3] #pos de la meilleure zone
    
    #return pente et origine, coeff correlation, pos, taille de la zone fittée
    
    return p, max_r, best_taille, best_pos
    
    
def demodulation(t,s, fexc, t1):
    # s signal de l'elevation (x,t)
    # # Exponentielle complexe pour la demodulation:
    c = np.mean(s*np.exp(-1j * 2 * np.pi * t[None,:] * fexc),axis=1)
    etademod = np.real(c[:,None]*np.exp(1j*2*np.pi*t1[ None,:]))
    return c, t1, etademod


    
        
    