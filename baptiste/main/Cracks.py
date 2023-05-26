# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:18:54 2022

@author: Louis Saddier

Détection Cracks, fait par Louis Saddier, modifié par Baptiste Auvity
"""

import matplotlib.pyplot as plt
import os
import baptiste.image_processing.cracks as cr



#%%
#### EXEMPLE #####  (pour une seule image)
path = "D:\Banquise\Baptiste\Resultats_video\img_skelet.tiff"
original_path = "D:\Banquise\Baptiste\Resultats_video\img_test.tiff"
crack, use, NX, NY = cr.cracks(path,5)
angles, weights = cr.analyze_cracks(crack, use, plot=True, original=original_path)
plt.show()
# histo_angles(angles,weights)
# plt.show()


#%% #### EXEMPLE #####  (pour un stack d'images)
#chemin du dossier contenant les images skeletonisées
pathdir = "D:/MSC/12_05_22/results/Acquis1_ske/"  #Ne pas oublier / à la fin
list_dir =  os.listdir(pathdir)

#chemin du dossier contenant les images originales (optionnel)
pathori = "D:/MSC/12_05_22/results/Acquis1_ori/"
list_ori = os.listdir(pathori)

# fréquence d'échantillonage des images
fech = 1.5     # Hz

# indique l'intervalle où l'on veut tracer l'histogramme
modulo = 100

for k in range(len(list_dir)):
    
    if k%modulo == 0:

        path = pathdir+list_dir[k]
        
        ori = pathori+list_ori[k]
        
        crack, use, NX, NY = cr.cracks(path, 7)
        angles, weights = cr.analyze_cracks(crack, use, plot=True, original=ori)
        plt.xlim(0,NX)
        plt.ylim(0,NY)
        plt.show()
        
        
        cr.histo_angles(angles,weights)
        plt.title('t = '+str(round(k/fech,1))+'s')
        plt.show()

        
        



