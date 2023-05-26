# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:53:47 2022

@author: Banquise
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:18:54 2022

Inspiré du code de Louis Saddier
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from skimage.morphology import skeletonize
import baptiste.image_processing.cracks as cr
import baptiste.files.file_management as fm
import baptiste.experiments.import_params as ip
import baptiste.display.display_lib as disp
import baptiste.image_processing.image_processing as imp
# %run parametres_FSD.py

#Va chercher les images et stock leur chemin dasn liste_images

path_images, liste_images, titre_exp = fm.import_images(loc,nom_exp,"FSD")

#Importe les paramètres de l'experience

facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = ip.import_param (titre_exp, date) 

mmparpixel = ip.import_calibration(titre_exp,date)            
              
# Creates a file txt with all the parameters of the experience and of the analysis
    

plot = True

display = True
one_image = True

save = False
saveplot = False

angle = False

temporel = True

scale = 255

crop = False
crop_haut = 205
crop_bas = 410
crop_droite = 30
crop_gauche = 105

size_subplot = 3

crack_lenght_min = 20 #longueur minimale de fissure en pixels
sepwb_cracks = 140   # separation blanc noir pour seuil Cracks
k_size_crack = 5
k_iteration_crack = 2


param_complets = ["Paramètres d'adimensionnement :", "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(k_size_crack), "kernel_iteration = " + str(k_iteration_crack) ,"nbframe = " + str(nbframe) , "minsize_crack = " + str(crack_lenght_min) , "sepwb = " + str(sepwb_cracks) , "size_crop = " + str(size_crop), "mmparpixel = " + str(mmparpixel), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp)]
param_complets.extend(["Paramètres d'affichage et traitement", "crop : " + str(crop),"temporel : " + str(temporel),"save : " + str(save),"saveplot : " + str(saveplot),"angle : " + str(angle),"one_image : " + str(one_image)])

#%% MAIN


if one_image : 
    display = True
    plot = False
    debut = 10
    nbframe = 1

if plot == True:
    display = False
    
    
temps = []
longueur_fissures = []

for k in range(debut, len (liste_images), int(len(liste_images)/ nbframe)):
    
    print('Analyzing frame '+ str(int (k / int(len(liste_images)/ nbframe)) + 1) + " over " + str(int(nbframe)))
    
    if temporel :
        temps.append(k / facq)
    else :
        temps.append(k)
        
    if display :
        disp.figurejolie()
        
    
    path = path_images+liste_images[k]
    
    image_originale = cv2.imread(path,0)
    
    if crop :       
        image_originale = image_originale[crop_haut :np.shape(image_originale)[0]-crop_bas , crop_gauche:np.shape(image_originale)[1] - crop_droite] 
    if display :
        disp.figurejolie()
        disp.joliplot('', '','', '', title = r'Image originale', image = image_originale)

        
    #binarisation de l'image
    
    img = image_originale < sepwb_cracks
    
    if display :
        disp.figurejolie()
        disp.joliplot('', '','', '', title = r'Image binarisée', image = img)

    
    img = imp.scale_256(img)
    img = np.invert(img)
    
    # On dilate l'image en vertical et horizontal
    
    kernel = np.ones((1,k_size_crack), np.uint8)
    
    img = cv2.dilate(img, kernel, iterations=k_iteration_crack)
    
    if display :
        disp.figurejolie()
        disp.joliplot('', '','', '', title = r'Image horizontale dilatée', image = img)

        
    
    kernel = np.ones((k_size_crack,1), np.uint8)
    
    img = cv2.dilate(img, kernel, iterations=k_iteration_crack)
    
    if display :
        disp.figurejolie()
        disp.joliplot('', '','', '', title = r'Image verticale dilatée', image = img)

    
    # img = np.invert(img)
    
    # #erode dialte pour enlever "la soupe" et laisser que les fissures
    # img = erodedilate(img, kernel_iteration, kernel_size, save, path_images, name_fig_crack)
    
    # img = np.invert(img)
    
    # if display :
    #     axes.append( fig.add_subplot(size_subplot, 2, u) )
    #     axes[-1].set_title("Image erode dilate")  
    #     plt.imshow(img, cmap=plt.cm.gray)
    #     plt.axis()
    #     u += 1
        
    img = img / 255
    #skeleton de l'image
    img_skeleton = skeletonize(img)
    
    if display :
        disp.figurejolie()
        disp.joliplot('', '','', '', title = r'Image skelétonée', image = img_skeleton)
        # plt.imsave("D:\Banquise\Baptiste\Resultats_video\img_skelet.tif" , img_skeleton)
        
    img = imp.scale_256(img_skeleton)
    
    #fonction craks: trouve les fissures, leur position et longueur
    crack, list_big_cracks, NX, NY = cr.cracks(img, crack_lenght_min)
    #analyze_craks trouve angle et longueur  des fisssures
    angles, weights = cr.analyze_cracks(crack, list_big_cracks, original=None)
    
    if display :
        disp.figurejolie()
        for i in range (len (list_big_cracks)):
            x = np.array(list_big_cracks[i])[:,1]
            y = np.array(list_big_cracks[i])[:,0]
            # plt.plot(x , y,color = 'red', ls = '-', lw = 0.2)
            disp.joliplot('', '',x, y, exp = False, color = 2)

        disp.joliplot('', '','', '', title = r'Image + fissures', image = image_originale)
        
        if save :
            plt.savefig(path_images[:-15] + "resultats/" + name_fig_crack + "image+fissures.tiff" , dpi = 300)
    longueur_fissures.append(np.sum(weights))
    # print ("Longueur totale des fissures : " + str(longueur_fissures * mmparpixel) + " mm ")
        
    if angle and display :
        disp.figurejolie()
        axes.append( fig.add_subplot(size_subplot, 2, u) )
        axes[-1].set_title("Angles")
        histo_angles(angles,weights)
        plt.title('t = '+ str(round(k/facq,1)) + 's')
        plt.show()
        

if plot :
    
    if temporel :
        disp.figurejolie()
        disp.joliplot('Temps (s)', 'mm', temps, longueur_fissures,legend = 'Taille totale des fissures', title = r'Evolution temporelle', log = True)
        
    else :
        disp.figurejolie()
        disp.joliplot('Frame number', 'mm', temps, longueur_fissures,legend = 'Taille totale des fissures', title = r'Evolution temporelle', log = True)
        

    if saveplot :
        
        plt.savefig(path_images[:-15] + "resultats" + "/" + name_fig_crack + "plot_temporel.tiff", dpi = 300)

        
param_complets = np.array(param_complets)
if save :
    np.savetxt(path_images[:-15] + "resultats" + "/Paramètres_" + str(name_fig_crack) + ".txt", param_complets, "%s")

#%% Fit log
plt.figure()

num_img = 40

p = np.polyfit( temps[-num_img:], longueur_fissures[-num_img:],1)
plt.plot(temps[-num_img:], longueur_fissures[-num_img:], 'ko', label = 'Taille totale des fissures')
polynome = np.zeros(len (temps[-num_img:]))
for i in range (len (polynome)):
    polynome[i] = temps[-num_img:][i] * p[0] + p[1]
plt.plot(temps[-num_img:], polynome,'mx', label = 'Fit linéaire ' + str(p[0]) + " * x " + str(p[1]))
plt.yscale('log')
plt.xscale('log')
plt.ylabel("mm")
plt.xlabel('Temps (s)')
plt.legend()
print (p)

