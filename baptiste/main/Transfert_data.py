# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:00:49 2024

@author: Banquise
"""

import shutil
import baptiste.files.file_management as fm
import os

path_ordi = 'E:\\Baptiste\\Resultats_video\\'
# path_ordi = 'D:\Banquise\\Baptiste\\Resultats_video\\'
# path_serveur = 'W:\Banquise\\Baptiste\\Resultats_exp\LAS\\'
path_serveur = 'H:\\Banquise\\Baptiste\\LAS\\'


liste_LAS = fm.get_files(path_ordi, 'LAS', '.npy')
liste_params = fm.get_files(path_ordi, 'las', '.pkl')

for u in range (len(liste_params)) :
    date = liste_params[u][29:35] #ATTENTION DIFFERENT SUR D [38:44] ET E [29:35]
    nom_exp = liste_params[u][44:49] #ATTENTION DIFFERENT SUR D [53:58] ET E [44:49]
    
    if float(date) > 231004 :
        try:
            os.mkdir(path_serveur + date)
        except:
            print(date)
        try:
            os.mkdir(path_serveur + date + '\\' + nom_exp)
        except:
            pass
        
        filePath = shutil.copy(liste_LAS[u], path_serveur + date + '\\' + nom_exp + '\\')
        filePath = shutil.copy(liste_params[u],path_serveur + date + '\\' + nom_exp + '\\')
    
    
    
# filePath = shutil.copy('file.txt', '/home/user/doc/')
# print(filePath)    #cela affiche /home/user/doc/file.txt filePath = shutil.copy(liste[0], '/home/user/doc/')