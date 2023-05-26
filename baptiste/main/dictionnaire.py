# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:02:18 2022

@author: Banquise

CREATION D'UN DICTIONNAIRE GLOBAL 

Permet de modifier des termes sur tout le dictionnaire entre deux dates indiquées. Permet aussi d'en créer des nouvelles,d'en enlever.

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



save = False
openn = True
redo = False
    
dico = {}
if openn :
    dico = open_dico()

loc = "D:\Banquise\Baptiste\Resultats_video\\"

date_min = 220411 #AVANT C'EST NUL
# date_min = 221005
date_max = 230120 #AJD ?

#%%Ajout de termes depuis un fichier txt ou csv
path_complement = 'D:\Banquise\Baptiste\Resultats\\complements_dictionnaires.txt'
# path_complement = 'D:\Banquise\Baptiste\Resultats\\complements_dico.csv'

# complements = np.loadtxt(path_complement, dtype = np.string_, delimiter = '\n', unpack = False) #open(path_complement, 'r')
complements = pandas.read_csv(path_complement, sep = '\t', header = None)
#date/nom_exp/CM(si 0 que prendre ?)/poids initila/poids final/analyse LAS/commentaire/casse
complements = pandas.DataFrame(complements).to_numpy()

lignes,colonnes = complements.shape

for i in range(1,lignes) :
    date = str(complements[i,0])
    nom_exp = complements[i,1]
    remove_dico(dico, date,nom_exp, 'angle_cam_LAS')
    for j in range (2,colonnes) : 
        dico = add_dico(dico,date,nom_exp,str(complements[0,j]),complements[i,j])
        
save = False
if save :
    save_dico(dico)


#%% Print de qlq aparametres

for date in dico :

    if float(date) > date_min and int(date) != 221011 :
        print(date)

        for nom_exp in dico[np.str(date)] :
            
            if 'amp_fracs_fft' in dico[date][nom_exp] :
                # if "non" in dico[date][nom_exp]['casse'] :
                print(nom_exp)

#%% Ajout de CM si CM par remesuré

for date in dico :
    if float(date) > date_min :
        print(date)
        for nom_exp in dico[np.str(date)] :
            nom_exp = str(nom_exp)
            if 'CM' in dico[date][nom_exp] :
                if float(dico[date][nom_exp]['CM']) < 0.0001 :
                    dico = add_dico(dico,date,nom_exp,'CM',dico['variables_globales']['CM']['CM'])
                    dico = add_dico(dico,date,nom_exp,'err_CM',dico['variables_globales']['CM']['err_CM'])
            dico = add_dico(dico, date,nom_exp, 'rho_utilise', 900)
                
                
                
#%% Ajout de h

for date in dico :
    
    if float(date) > date_min and int(date) != 221011 :
        print(date)
        
        for nom_exp in dico[np.str(date)] :
            dico = add_dico(dico, date,nom_exp, 'rho_utilise', 900)
            if 'poids_initial' in dico[date][nom_exp] :
                if float(dico[date][nom_exp]['poids_initial']) > 0.0001 :
                    nom_exp = str(nom_exp)
                    poids = float(dico[date][nom_exp]['poids_initial']) - float(dico[date][nom_exp]['poids_final'])
                    surface =  float(dico[date][nom_exp]['Larg_ice']) * float(dico[date][nom_exp]['Long_ice'])
                    rho = float(dico[date][nom_exp]['rho_utilise'])
                    CM = float(dico[date][nom_exp]['CM'])
                    h = poids/(surface*rho*CM)
                    if "err_CM" in dico[date][nom_exp] :
                        err_CM = float(dico[date][nom_exp]['err_CM'])
                        err_h = err_CM/CM * h
                        dico = add_dico(dico,date,nom_exp,'err_h',err_h )
                    dico = add_dico(dico, date, nom_exp, 'h', h)





#%% Ouverture tt les exp pour enregistrer les paramètres

if redo :
    fichiers = os.listdir(loc) 
    
    a = [] #LISTE EXP totale
    
    for i in fichiers :
        a.append(os.listdir(loc + i))
        
        
    for j in a : #pr une date
        date =j[0][1:7]
        
        if date not in dico :
            dico[date] = {}
        loc_date = loc + 'd' + date + '\\'
        for k in j : #pr une exp de cette date    
            if k[0] == 'd' :
                if date_max >= float(date) and float(date) > date_min :
                    
                    liste_param = []
                    name_param = []
                    nom_exp = k[8:13]
                    type_exp = k[14:17]
                    if "image_sequence" in os.listdir(loc_date + k) :
                        path_images, liste_images, titre_exp = import_images(loc_date,nom_exp,type_exp)
                        liste_param.append(len(liste_images))
                        name_param.append('nb_frame')
                    else :
                        titre_exp = k
                    if float(date) >= 221020 :
                        liste_param.extend(import_param(titre_exp,date))
                        name_param.extend(['facq', 'texp', 'fexc', 'Vmot', 'Hw', 'Larg_ice', 'Long_ice', 'tacq', 'type_exp'])
                    else :
                        liste_param.extend(import_param(titre_exp,date))
                        name_param.extend(['facq', 'texp', 'Tmot', 'Vmot', 'Hw', 'Larg_ice', 'Long_ice', 'tacq', 'type_exp'])
                    
                    if type_exp == 'LAS' :
                        liste_param.extend(import_calibration(titre_exp,date))
                        name_param.extend(['mmparpixelx', 'mmparpixely', 'mmparpixelz', 'angle_cam_LAS', 'mmparpixel'])
                    if type_exp == 'FSD' :
                        liste_param.append(import_calibration(titre_exp,date))
                        name_param.extend(['mmparpixel'])
                    if type_exp == 'PIV' :
                        liste_param.append(import_calibration(titre_exp,date))
                        name_param.extend(['mmparpixel'])
                    if type_exp == 'IND' :
                        liste_param.append(import_calibration(titre_exp,date))
                        name_param.extend(['mmparpixel'])
                    
                    if float(date) >= 221006 :
                        cam_dessus = True
                        if type_exp == "LAS" :
                            if "references" in os.listdir(loc_date + k) :
                                if 'angle_laser.txt' in os.listdir(loc_date + k + '\\references') :
                                    liste_param.extend(import_angle(date, nom_exp, loc_date))
                                    liste_param.append(cam_dessus)
                                    name_param.extend(['grossissement', 'er_grossissement', 'angle', 'er_angle','cam_dessus'])
                    else :
                        cam_dessus = False 
                        liste_param.extend([cam_dessus])
                        name_param.extend(['cam_dessus'])
                    
                    if nom_exp not in dico[date] :
                        dico[date][nom_exp] = {}
                    for uuu in range (len(name_param)):
                        if name_param[uuu] not in dico[date][nom_exp] :
                            dico[date][nom_exp][name_param[uuu]] = liste_param[uuu]
                            if name_param[uuu] == 'Larg_ice' :
                                dico[date][nom_exp][name_param[uuu]] = liste_param[uuu]/100                          
                            if name_param[uuu] == 'Long_ice' :
                                dico[date][nom_exp][name_param[uuu]] = liste_param[uuu]/100
                    if dico[date][nom_exp]['CM_final'] == 0 :
                        dico[date][nom_exp]['CM'] == 0
                        dico[date][nom_exp]['CM_final'] == 0
                               
                    # if nom_exp[-1] == 'R' :
                    #    if float(date) >= 230103 :
                    #       D = dico[date][nom_exp]["D"]
                    #       Ld = dico[date][nom_exp]["Ld"]
                    #       grossissement = dico[date][nom_exp]["grossissement"]
                    #       rho_utilise = dico[date][nom_exp]["rho_utilise"]
                    #       cam_dessus = dico[date][nom_exp]["cam_dessus"]
                    #       angle = dico[date][nom_exp]["angle"]
                    #       er_angle = dico[date][nom_exp]["er_angle"]
                    #       er_grossissement = dico[date][nom_exp]["er_grossissement"]
                    #       for nom_exp in dico[date] :
                    #           dico = add_dico(dico,date,nom_exp,'Ld',Ld)
                    #           dico = add_dico(dico,date,nom_exp,'D',D)
                    #           dico = add_dico(dico,date,nom_exp,'grossissement',grossissement)
                    #           dico = add_dico(dico,date,nom_exp,'rho_utilise',rho_utilise)
                    #           dico = add_dico(dico,date,nom_exp,'cam_dessus',cam_dessus)
                    #           dico = add_dico(dico,date,nom_exp,'angle',angle)
                    #           dico = add_dico(dico,date,nom_exp,'er_angle',er_angle)
                    #           dico = add_dico(dico,date,nom_exp,'er_grossissement',er_grossissement)
                        
                                
                    
                        
                            

                    
              
if save : 
    save_dico(dico)

            
          




