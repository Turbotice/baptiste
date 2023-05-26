# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:32:14 2023

@author: Banquise
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:38:16 2022

@author: Turbots
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
import os
from PIL import Image
%run parametres_FSD.py
%run Functions_FSD.py
%run display_lib_gld.py


path_images, liste_images, titre_exp = import_images(loc,nom_exp,"LAS")

#Importe les paramètres de l'experience

facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = import_param (titre_exp,date)  

mmparpixel = import_calibration(titre_exp,date)           
          
# Creates a file txt with all the parameters of the experience and of the analysis

openn_dico = True
if openn_dico :
    dico = open_dico()


param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(kernel_size), "kernel_iteration = " + str(kernel_iteration) ,"nbframe = " + str(nbframe) ,"maxsize = " + str(maxsize) , "minsize = " + str(minsize) , "threshold = " + str(threshold) , "sepwb = " + str(sepwb) , "size_crop = " + str(size_crop) , "bordersize = " + str(bordersize), "mmparpixel = " + str(mmparpixel), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp)]
mmparpixel = dico[date][nom_exp]['mmparpixel'] #0.2196
      
#%% Initialisation


plot = False         #pour plot l'evolution temporelle

display = True     #si on veut afficher les resultats pour chaque image
histogram = False  #si on veut faire un histogram des distribution de taille (1 par image)
img_debase = True  #si on veut afficher l'image de départ

crop = False        #pour cadrer l'image
crop_haut = 205
crop_bas = 410
crop_droite = 30
crop_gauche = 105

temporel = False    #pour faire un graph en temps (si False, en numero de frame)

size_subplot = 2 

save = False        #pour sauvegarder les images traitées
saveplot = False     #pour sauvegarder le plot temporel
savefigure = False    #pour sauvegarder le subplot (avec FSD et les images, besoin de display pour marcher)
save_param = False   #Pour sauvegarder les paramètres dans un fichiers texte
save_data = False

if nbframe > len(liste_images):
    nbframe = len(liste_images)

#%% MAIN

if plot == True :
    histogram = False
    display = False
    img_debase = False


# param_complets.extend(["Paramètres d'affichage et traitement", "crop : " + str(crop),"binaire : " + str(binaire),"temporel : " + str(temporel),"save : " + str(save),"saveplot : " + str(saveplot)])



liste_aire = []
liste_N = []
taille_mediane = []
taille_moyenne = []

temps = []
premiere_boucle = True




for i in range (debut, len (liste_images), int(len(liste_images)/ nbframe + 1)):
    
    
    airetot = 0
    airetot1 = 0
    u = 1
    
    if display :
        axes, fig = figurejolie(subplot = (size_subplot,2))
        
    if temporel :
        temps.append(i / facq)
    else :
        temps.append(i)

    # Loading of the image to analyze
    
    image_originale = cv2.imread (path_images + liste_images[i], cv2.IMREAD_GRAYSCALE)
    
    #If you want to crop the pictures
    
    if crop :       
        image_originale = image_originale[crop_haut :np.shape(image_originale)[0]-crop_bas , crop_gauche:np.shape(image_originale)[1] - crop_droite] 
    
    # Adding white edges around the image
       
    image_gray = cv2.copyMakeBorder(image_originale, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
    imsize = np.shape(image_gray)
    
    #Displays the originale image with black borders

    if display and img_debase :
        figurejolie()
        joliplot('', '', (), (), image = image_gray, title = 'Image grise')
        # axes = joliplot('', '', [], [], color = False, fig = fig, axes = axes, title = "Image " + str(i) + " / " + str(len(liste_images)), subplot = (size_subplot,2), legend = False, image = image_gray)
        plt.axis('off')
    
    print('Analyzing frame '+ str(int (i / int(len(liste_images)/ nbframe)) + 1) + " over " + str(int(nbframe)))
    
    #Contrast enhancement
    
    I_min, I_max = np.min(image_gray),np.max(image_gray)
    image_gray = ((image_gray-I_min) / (I_max-I_min) * scale)
   
    #Detection by threshold
        

        
    image_binaire = image_gray < sepwb
        
    image_binaire = (image_binaire * scale).astype(np.uint8)
        
    if save :
        plt.imsave(path_images[:-15] + "resultats/" + name_fig + "image_binarisée.tiff" , image_binaire, cmap=plt.cm.gray )
            
    image_binaire = erodedilate(image_binaire, kernel_iteration, kernel_size, save, path_images, name_fig)
    
    image_binaire = cv2.bitwise_not(image_binaire)
    
    
     
    if display :
        
        figurejolie()
        joliplot('', '', (), (), image = image_binaire, title = 'image binarisée')
        
        # axes = joliplot('', '', [], [], color = False, fig = fig, axes = axes, title = "Image binaire, threshold = " + str (sepwb) + "_" + str (kernel_size) + "_" + str(kernel_iteration), subplot = (size_subplot,2), image = image_gray)
        plt.axis('off')
        
        

    #detection and storage of the borders

    diametres = []
    FSD = []
    contours_binaires, hierarchy = cv2.findContours(image_binaire,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_EXTERNAL pr que les contours externes (marche pas)
    img_contours_binaire = np.zeros((imsize[0],imsize[1]), np.uint8)
        
    for j in range (len (contours_binaires)):
            
        # print ("étape " + str( j) + " sur " + str( len(contours)))
            
        area = cv2.contourArea(contours_binaires[j])
        airetot += area
        diametres.append (np.sqrt(4*area/np.pi))
            
        #we keep the borders of the size between minsize and maxsize
            
        if diametres[j] >= minsize and diametres[j] <= maxsize:
            cv2.drawContours(img_contours_binaire,contours_binaires,contourIdx = j, color = (255,255,255), thickness = 2)
            FSD.append( diametres[j])
            
    #creation of lists with the air, median, average, number of fragments
    if premiere_boucle :

        N0 = len(FSD) + 1
        aire_glace = airetot * mmparpixel**2

 
    liste_N.append (len (FSD)/N0)
    liste_aire.append( airetot* mmparpixel**2 /aire_glace )
    taille_mediane.append(np.median(FSD) * mmparpixel/ lambda_vague)
    taille_moyenne.append(np.mean(FSD) * mmparpixel/ lambda_vague)


    if display :
        
        figurejolie()
        joliplot('', '', (), (), image = img_contours_binaire, title  = "contours")
        
        # axes = joliplot('', '', [], [], color = False, fig = fig, axes = axes, title = "Contours entre " + str (int (minsize * mmparpixel)) + " et " + str (int (maxsize * mmparpixel)) + "mm" + " img_contours_binaire", subplot = (size_subplot,2), image = img_contours_binaire)
        plt.axis('off')
        
        if save :
            plt.imsave(path_images[:-15] + "resultats/" + name_fig + str (kernel_size) + "_" + str(kernel_iteration) +"img_contours_binaire.tiff" , img_contours_binaire, cmap=plt.cm.gray )
            
  

    #conversion from pixels to mm

    for w in range (len(FSD)):
        FSD[w] = FSD[w] * mmparpixel
    
    if histogram :
        nb_boites = len (FSD) # 1000 #int(((len (FSD))) ** 0.6 * 2)
        
        [n, x] = np.histogram(FSD,nb_boites)
        #evaluate the cumulative
        cumulative = np.cumsum(n)

        
        
        # [n,x]=np.histogram(FSD,nb_boites, cumulative = True, histtype='step')
        xc= (x[1:]+x[:-1]) / 2
    
    if display and histogram:
        
        figurejolie()
        joliplot(r'Fragment size (mm)', r"Fragment amount" , xc, len (xc) - cumulative, color = 7, title = r"Fragement size distribution, N = " + str (len(FSD)), log = True, exp = True)
        
        # axes = joliplot(r'Fragment size (mm)', r"Fragment amount" , xc, len (xc) - cumulative, color = 2,fig = fig, axes = axes, title = "Fragement size distribution, N = " + str (len(FSD)),subplot = (size_subplot,2), log = True, exp = True)

        if savefigure :
            plt.savefig(path_images[:-15] + "resultats/" + name_fig + "_" + str(i) + "_" + str (kernel_size) + "_" + str(kernel_iteration) +  "_FSD_binaire.tiff" )
    
    if display  :
        contourssuroriginale = np.zeros(np.append(image_gray.shape,[3]), dtype = 'uint8')
        contourssuroriginale[:,:,2] = np.asarray(image_gray, dtype = 'uint8')
        contourssuroriginale[:,:,1] = np.asarray(image_gray, dtype = 'uint8')
        contourssuroriginale[:,:,0] = img_contours_binaire
        figurejolie()
        joliplot("","","","",image = contourssuroriginale)
    
    cv2.destroyAllWindows()

if display: 
    plt.show()

# temps = np.append (temps[10:16],temps[40:])
# liste_N = np.append (liste_N[10:16],liste_N[40:])
# liste_aire = np.append (liste_aire[10:16],liste_aire[40:])
# taille_moyenne = np.append (taille_moyenne[10:16],taille_moyenne[40:])
# taille_mediane = np.append (taille_mediane[10:16],taille_mediane[40:])


if plot :
    figurejolie()
    # plt.plot (temps, liste_N, label = 'N / N0 (' + str(N0) + ')',marker = 'x')
    # plt.plot (temps, liste_aire, label = 'Aire / Aire glace 0',marker = 'x')
    # plt.plot (temps, taille_moyenne, label = 'Moyenne / Longueure d\'onde vague',marker = 'x')
    # plt.plot (temps, taille_mediane, label = 'Médiane / Longueure d\'onde vague',marker = 'x')
    # plt.ylabel("Unité Arbitraire")
    if temporel :
        xlabel = r'Temps (s)'
    else :
        xlabel = r'Frame number'
        
    joliplot(xlabel, r"Arbitrary unit" , temps, liste_N, color = 2,legend = 'N / N0 (' + str(N0) + ')', title = r"Temporal evolution", log = True, exp = False)
    joliplot(xlabel, r"Arbitrary unit" , temps, liste_aire, color = 3,legend = 'Aire / Aire glace 0', title = r"Temporal evolution", log = True, exp = False)
    joliplot(xlabel, r"Arbitrary unit" , temps, taille_moyenne, color = 6,legend = 'Moyenne / Longueure d\'onde vague', title = r"Temporal evolution", log = True, exp = False)
    joliplot(xlabel, r"Arbitrary unit" , temps, taille_mediane, color = 8,legend = 'Médiane / Longueure d\'onde vague', title = r"Temporal evolution", log = True, exp =False)
    
    
#%% SAVE
xc = np.asarray(xc)
data_histo = np.c_[xc, len (xc) - cumulative]

if save_data :
    np.savetxt(path_images[:-15] + "resultats/" + name_fig + "data_histo_distribution.txt", data_histo, "%s")

if saveplot :
    plt.savefig(path_images[:-15] + "resultats" + "/" + name_fig + "plot_temporel.tiff")
if save_param: 
    param_complets = np.array(param_complets)
    np.savetxt(path_images[:-15] + "resultats/" + name_fig + "Paramètres.txt", param_complets, "%s")