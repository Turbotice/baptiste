# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:53:07 2023

@author: Antonin

Prend les données de cityshark et les copie sur le serveur

de C:\CityShark

à W:\Banquise\Rimouski 2023\Data\Geophones\AAAAMMJJ

En mettant chaque fichier dans le dossier à la bonne date (le crée si besoin)
Et en les mettant en .txt à la fin (que si pas .txt) (sur les deux endroits)

"""



import glob
import os
import shutil

path_cityshark = 'C:\CityShark\\'

path_save = 'W:\Banquise\Rimouski 2023\Data\Geophones\\'

liste_fichiers = glob.glob(path_cityshark + '*')

for i in liste_fichiers :
    if i[-4:] != ".txt" :
        os.rename( i, i + ".txt" )
 
liste_fichiers = glob.glob(path_cityshark + '*')

for i in liste_fichiers :
    
    name_file = i[-19:-4]
    date_file = name_file[:6]
    
    
    if os.path.isdir(path_save + "20" + date_file ) == False:
        os.mkdir(path_save + "20" + date_file )
        
    shutil.copy(i , path_save  + "20" + date_file + "\\" +  name_file + ".txt")
        
    
        
    

