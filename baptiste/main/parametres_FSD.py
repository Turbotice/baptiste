# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:31:31 2022

@author: Turbots
"""

import numpy as np
import os

# PARAMETRES EXPERIENCE
date = "221124"
nom_exp = "NBFT1" #nom de l'experience en 5 lettres qui la caractérise
loc = "D:\Banquise\Baptiste\Resultats_video\d" + date + "\\"
loc_resultats = "D:\Banquise\Baptiste\Resultats"


name_fig = date + "_" + nom_exp + "_FSD_"
name_fig_LAS = date + "_" + nom_exp + "_LAS_"
name_fig_crack = date + "_" + nom_exp + "_CRACK_"
name_fig_FFT = date + "_" + nom_exp + "_lambda_kappa_demod_"
name_fig_IND = date + "_" + nom_exp + "_IND_"
path_results = 'D:\Banquise\Baptiste\Résultats\\' + 'FSD\\' + nom_exp


'''
PARAMETRES TRAITEMENT IMAGE
'''

debut = 30000
nbframe = 1      #nb de frame que l'on veut regarder à intervalle de temps régulier
nbplot = 3           #le nombre de plot complet qu'on veur


#Pour faire des variables sans dimension


lambda_vague = 10 #cm
Ampvague = 0.2 #cm
#viendra de LAS plus tard


'''
                    FSD
'''

kernel_size = 5         #Pour erodedilate
kernel_iteration = 3

maxsize = 100000      #en pixel, taille min et max de fragments
minsize = 10

sepwb = 200         # separation blanc noir pour seuil FSD

size_crop = 50
bordersize = 20


#

'''
                    LAS
'''

med_size = 1
k_size = 5
k_iteration = 3
sigma = 150         #nb pr la fct de convolution (largeur environ, 30 de large pour sigma = 150)




#import les paramètres de l'exp avec le titre du fichier de l'exp

def import_param (titre_exp,date, exp_type = "TT"):
    if exp_type == "TT":
    #"\d" + date + "_vagues_facq" + str(facq) + "Hz_texp" + str(texp) + "us_fmotor"+ str(fpot) + "_Amplitude" +str(Apot) + "_Hw" + str(Hw) + "cm_Pasdamier" + str(pasdamier) + "mm"
        if float(date) >= 221020 :  
            facq = float ( titre_exp[titre_exp.index("facq") + 4:titre_exp.index("facq") + 7]) # fréquence d'acquisition (Hz)
        else :
            facq = float ( titre_exp[titre_exp.index("facq") + 4:titre_exp.index("facq") + 6]) # fréquence d'acquisition (Hz)
        texp = float ( titre_exp[titre_exp.index("texp") + 4:titre_exp.index("texp") + 8])# temps d'exposition (muS)
        Tmot = float ( titre_exp[titre_exp.index("Tmot") + 4:titre_exp.index("Tmot") + 7])# tps entre mvt moteur (ms)
        Vmot = float ( titre_exp[titre_exp.index("Vmot") + 4:titre_exp.index("Vmot") + 7])# vitesse du moteur (0 - 255)
        Hw = float ( titre_exp[titre_exp.index("Hw") + 2:titre_exp.index("Hw") + 4])# hauteur eau (cm)
        if titre_exp[titre_exp.index("sur") + 5].isdigit() :
            Long_ice = float ( titre_exp[titre_exp.index("sur") + 3:titre_exp.index("sur") + 6])#longeur glace (cm)
        else :
            Long_ice = float ( titre_exp[titre_exp.index("sur") + 3:titre_exp.index("sur") + 5])#longeur glace (cm)
        Larg_ice = float ( titre_exp[titre_exp.index("sur") - 2:titre_exp.index("sur")])#largeur glace (cm)
        
        tacq =  float ( titre_exp[titre_exp.index("tacq") + 4:titre_exp.index("tacq") + 7])#temps total acquisition (s)
        type_exp =  str (titre_exp[14:17])    #FSD pour cassage glace, FCD pour mesure damier, LAS pour profil laser
        return facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp
    if exp_type == "IND":
        h = float ( titre_exp[titre_exp.index("ha") + 2:titre_exp.index("ha") + 4])
        return h





