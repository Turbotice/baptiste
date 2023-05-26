# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:36:42 2022

@author: Banquise
"""

"""Comptage de fichiers dans image sequence dans tous les dossiers"""


import os
import numpy as np


loc = "D:\Banquise\Baptiste\Resultats_video\\"
date_min = 230119

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
                                
#%%
import numpy as np

liste_images = os.listdir("D:\Banquise\Baptiste\Resultats_video\d221202\d221202_TTTTT\image_sequence")
uu = len(liste_images)
if uu > 9 :
    print(uu)
    print(k[:13])
    liste_len = []
    for lll in liste_images :
        liste_len.append(len(lll))
    for mmm in range(len(liste_images)) :
        if liste_len[mmm] < np.max(liste_len) :
            os.rename("D:\Banquise\Baptiste\Resultats_video\d221202\d221202_TTTTT\image_sequence\\" + liste_images[mmm], "D:\Banquise\Baptiste\Resultats_video\d221202\d221202_TTTTT\image_sequence\\" +liste_images[mmm][:-9] + '0' +liste_images[mmm][-9:] )
                
# os.listdir("D:\Banquise\Baptiste\Resultats_video\d221202\d221202_TTTTT\image_sequence")[mmm][:-9] + '0' + os.listdir("D:\Banquise\Baptiste\Resultats_video\d221202\d221202_TTTTT\image_sequence")[mmm][-9:]
   