# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:36:42 2022

@author: Banquise
"""

"""Comptage de fichiers dans image sequence dans tous les dossiers"""


import os
import numpy as np


loc = "D:\Banquise\Baptiste\Resultats_video\\"
date_min = 231110

fichiers = os.listdir(loc) 

a = []

for i in fichiers :
    a.append(os.listdir(loc + i))
    
    
for j in a :
    for k in j :
        if k[0] == 'd' :
            if float(k[1:7]) > date_min :
                if "image_sequence" in os.listdir(loc + k[:7] + "\\" + k) :
                    liste_images = os.listdir(loc + k[:7] + "\\" + k + "\image_sequence")
                    uu = len(liste_images)
                    if uu > 9999 :
                        print(uu)
                        print(k[:13])
                        liste_len = []
                        for lll in liste_images :
                            liste_len.append(len(lll))
                        for mmm in range(len(liste_images)) :
                            if liste_len[mmm] < np.max(liste_len) :
                                os.rename(loc + k[:7] + "\\" + k + "\image_sequence\\" + liste_images[mmm], loc + k[:7] + "\\" + k + "\image_sequence\\" + liste_images[mmm][:-9] + "0" + liste_images[mmm][-9:])
                                
#%% ATTENTION NE MARCHE PAS
import numpy as np
import os

path = 'D:\Banquise\\Baptiste\\Resultats_video\\d231013\\exp_analogue_geo\\test_2_13_10_23\\images_1_10000'

liste_images = os.listdir(path)
uu = len(liste_images)
if uu > 99999 :
    print(uu)

    liste_len = []
    for lll in liste_images :
        liste_len.append(len(lll))
    for mmm in range(len(liste_images)) :
        os.rename(path + "\\" + liste_images[mmm], path + "\\" + liste_images[mmm][:-12] + liste_images[mmm][-11] + '00' + liste_images[mmm][-10:] )
        
        
        # if liste_len[mmm] == (np.max(liste_len) - 2):
            
    #         os.rename(path + "\\" + liste_images[mmm], path + "\\" + liste_images[mmm][:-11] + '0' +liste_images[mmm][-11:] )

    #     if liste_len[mmm] == (np.max(liste_len) - 3) :

    #         os.rename(path + "\\" + liste_images[mmm], path + "\\" + liste_images[mmm][:-10] + '00' +liste_images[mmm][-10:] )
            
    #     if liste_len[mmm] == (np.max(liste_len) - 4) :

    #         os.rename(path + "\\" + liste_images[mmm], path + "\\" + liste_images[mmm][:-9] + '000' +liste_images[mmm][-9:] )
                
# os.listdir("D:\Banquise\Baptiste\Resultats_video\d221202\d221202_TTTTT\image_sequence")[mmm][:-9] + '0' + os.listdir("D:\Banquise\Baptiste\Resultats_video\d221202\d221202_TTTTT\image_sequence")[mmm][-9:]
   