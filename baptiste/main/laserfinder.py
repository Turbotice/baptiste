# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:08:41 2022

@author: Turbots
"""

#%%IMPORTATION DES DONNEES


import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
from scipy.signal import convolve2d
from scipy.signal import medfilt2d
from scipy import ndimage
import os
from PIL import Image
%run parametres_FSD.py
%run Functions_FSD.py
%run display_lib_gld.py


#Va chercher les images et stock leur chemin dasn liste images

path_images, liste_images, titre_exp = import_images(loc, nom_exp, "LAS")

#Importe les paramètres de l'experience



facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = import_param (titre_exp, date)    

mmparpixelx, mmparpixely, mmparpixelz, angle_cam_LAS, mmparpixel = import_calibration(titre_exp,date)          
              
# Creates a file txt with all the parameters of the experience and of the analysis
    
param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(kernel_size), "kernel_iteration = " + str(kernel_iteration) ,"nbframe = " + str(nbframe) ,"maxsize = " + str(maxsize) , "minsize = " + str(minsize) , "threshold = " + str(threshold) , "sepwb = " + str(sepwb) , "size_crop = " + str(size_crop) , "bordersize = " + str(bordersize), "mmparpixel = " + str(mmparpixel), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp)]

if nbframe > len(liste_images):
    nbframe = len (liste_images)
#%%PARAMETRES
MODE = "test"
# MODE = "go"

haut_LAS = 470#700 CCCS2#820 #RCSC1 780 CCCS1#260 RBT01
bas_LAS = 250#500 CCCS2#570 # 380#180 RBT01

display = True
display_result_only = True


gradient = False
maximum = True
red_ratio = False
convconv = True

moyenner = False    #faire un moyennage local (convolve2d)
medfilter = False    #mettre un filtre médian et dilate horizontal

save = False
saveplot = False


if MODE == "test" :
    display = True
    save = False
    nbframe = 3
    
if MODE == "go" :
    display = False
    save = True
    nbframe = len (liste_images)
    # debut = 7850
    



#%%CONVOLUTION

if display and nbframe > 20 :
    continuer = input ("Attention plus de 20 figures à afficher, continuer ?")
    if not (continuer == "oui") :
        exit()

if convconv :
    maximum = True



size_conv = 15
precision_conv = 40
xf = np.linspace(-size_conv,size_conv,precision_conv)
sigma = 70
fconv = np.exp(-xf**2/(2*sigma))
# fconv = (1-xf**2/sigma)*np.exp(-xf**2/(2*sigma))
fconv = fconv - np.mean(fconv)
if False :
    figurejolie()
    plt.plot(xf, fconv)



debut_zone = bas_LAS - size_conv * 0
fin_zone = haut_LAS + size_conv * 0

im = cv2.imread(path_images + liste_images[0])
im_2 = im[debut_zone:fin_zone,:,2]
y = np.zeros((len(liste_images),(np.shape(im_2)[1])))


#%%TRAITEMENT PRINCIPAL
for i in range (debut, len (liste_images), int(len(liste_images)/ nbframe )) :
    
    #Ouverture image et choix du canal Rouge
    
    im = cv2.imread(path_images + liste_images[i], cv2.IMREAD_UNCHANGED)
    
    # SELECTIONNE LA REGION D'INTERET
    
    im = np.array(im[debut_zone:fin_zone,:,:])
    I_min, I_max = np.min(im), np.max(im)
    im = ((im-I_min) / (I_max-I_min) * 255).astype(np.uint8)
    im_red = im[:,:,2].copy()
    # I_min, I_max = np.min(im_red), np.max(im_red)
    # im_red = ((im_red-I_min) / (I_max-I_min) * 255).astype(np.uint8)
    param1 = 30
    param2 = 30
    param3 = 16
    dst = cv2.fastNlMeansDenoising(im_red,None,param1,param2,param3)
    
    
    
    if display and not display_result_only :
        figurejolie()
        joliplot('', '', (), (), image = im_red, title = "Image rouge")
    
    im_las = np.zeros(((np.shape(im_red)[0]),(np.shape(im_red)[1])), dtype = np.uint8)
    im_las_base = np.zeros(((np.shape(im_red)[0]),(np.shape(im_red)[1])), dtype = np.uint8)
    
    if moyenner :        
        #Convolution pour smoother l'image  
        siz = 5
        
        im_red = convolve2d (im_red, np.ones((siz,siz))/ siz ** 2 , mode = 'same')
        
    if medfilter :
        #filtre médian et dilate en horizontal
        im_red_traitee = medfilt2d(im_red, med_size)
        
        if display and not display_result_only:
            figurejolie()
            joliplot('', '', (), (), image = im_red_traitee, title = "Image filtre médian " + str (med_size) )
        
        k_size = 8
        k_iteration = 4 #OPTI pour laser sur le téco
        
        kernel = np.ones((1,k_size), np.uint8)
        
        im_red_traitee = cv2.dilate(im_red_traitee, kernel, iterations=k_iteration)
        
        if display and not display_result_only :
            figurejolie()
            joliplot('', '', (), (), image = im_red_traitee, title = "Image dialted size " + str (k_size) + " iterations " + str(k_iteration))
            
    else :
        im_red_traitee = dst # im_red
        if display and not display_result_only:
            figurejolie()
            joliplot('', '', (), (), image = im_red_traitee, title = "Image filtre denoise " + str (param1) + " " + str (param2) + " "+ str (param3))
     
    [nx,ny] = np.shape(im_red_traitee)
    
    
    for k in range (ny): # colonne
    
    # DETECTION MAX GRADIENT
        if convconv : 
            xXxtentacion = np.arange(0,nx)
            im_conv = np.convolve(im_red_traitee[:,k], fconv, mode = 'same' )
            
            if display and  (k == 200 or k == 700 or k == 1500) and not display_result_only :
                figurejolie()
                joliplot('', '', xXxtentacion, im_red_traitee[:,k], title = "Convolve avec profil " + str(k) + ' ieme colonne', exp =False) 
                joliplot('', '', xXxtentacion, im_conv, title = "Convolve avec profil " + str(k) + ' ieme colonne', exp = False)
            
        if gradient :
            
            difCol = np.gradient(im_red_traitee[:,k])
            
            Imax = max(difCol[20:-40])
            indice1 = np.where(difCol == Imax)
            indice = indice1 [0][0]
            
            y0 = indice #pk + 6 ?
            
            a = 5
            
            z = difCol[y0-a:y0+a]
            
            x = [u for u in range (y0-a,y0+a) - y0 ]
            
            p = np.polyfit(x,z,2)    #on peut utiliser fminsearch pour fitter par une fonction quelconque
            
            imax = -p[1]/(2*p[0])
            
            y[i,k] = indice + imax
            
            #creation d'une image pour la superposition
            if y0 + imax > 0 and y0 + imax < 800and display:
                im_las [int(y0 + imax)-1:int(y0 + imax)+1,k] = 255
            
              #DETECTION MAX INTENSITE 
              
        if red_ratio :
            # if np.mod(k,100)==0:
            #     plt.figure()
            #     plt.plot(im2[:,k,1],'go')
            #     plt.plot(im2[:,k,0],'bo')
            #     plt.plot(im2[:,k,2],'ro')
                
            #     I = np.sum(im2[:,k,:],axis=1)
            #     plt.plot(I/3,'k+')
                
            #     plt.plot((im2[:,k,2]-im2[:,k,1]-im2[:,k,0]),'ms')
                    
            #[Imax,indice] = max(diff(im_2(5:end,k)));
            Imax = max(im2 [:,k,2] - (im2 [:,k,1] + im2 [:,k,0] ))
            indice = np.argmax(im2 [:,k,2] - (im2 [:,k,1] + im2 [:,k,0] ))
            #indice = indice1 [0][0]
            y0 = indice
            
            a=5
            
            imin = max(1,indice-a)
            imax = min(nx,indice+a)
              
            z = im_red_traitee[imin:imax,k]
            x = [ u for u in range (imin-indice,imax-indice)]
                
            p = np.polyfit(x,z,2)  #on peut utiliser fminsearch pour fitter par une fonction quelconque
                
            imax = -p[1]/(2*p[0])
            y[i,k] = y0 + imax
            #creation d'une image pour la superposition
            if y0 + imax > 0 and y0 + imax < 800 and display:
                im_las [int(y0)-1:int(y0)+1,k] = 255    
                
                 
        if maximum :
            
            if convconv :
                colonne_k = im_conv
            else :
                colonne_k = im_red_traitee[:,k]
               
            #trouver le max
            border = int(precision_conv/2)

            indice =  np.argmax(colonne_k[border:-border])
            indice = indice + border
              
            if display:
                im_las_base[indice,k] = 255 
            
            #sub_pixellaire
            
            a=4 #zone pour le fit polynome
            
            
            
            
            imin = max(1,indice-a)
            imax = min(nx,indice+a)
              
            z = colonne_k[imin:imax]
            x = [ u for u in range (imin-indice,imax-indice)]
                
            p = np.polyfit(x,z,2)  #on peut utiliser fminsearch pour fitter par une fonction quelconque
                
            imax = -p[1]/(2*p[0])
            y[i,k] = indice + imax
            
            
            #creation d'une image pour la superposition
            if indice + imax > 0 and indice + imax < nx and display:
                im_las[int(indice + imax) - 1:int(indice + imax) + 1,k] = 255 
                
                
    #      MONTRE LA DETECTION EN DIRECTE SUPERPOSEE A L'IMAGE BRUTE  
    if display:  #length(x)<11
        im_display = np.zeros(im.shape, dtype = np.uint8)
        im_display[:,:,2] = im[:,:,2]   #Image de base
        im_display[:,:,1] = im_las_base #Laser detecté
        im_display[:,:,0] = im_las      #Laser detecté subpixellaire
        if display :
            figurejolie()
            joliplot('', '', (), (), image = im_display, title = "Image " + str(i) + " / " + str(len(liste_images) ))
            plt.axis()
        
        if save :
            plt.imsave(path_images[:-15] + "resultats" + "/image_superposée_wow.tiff" , im_display)   

    
    if np.mod(i,100)==0:
        print('On processe l image numero: ' + str(i) + ' sur ' + str(len(liste_images)))
    
    if saveplot :
        plt.savefig(path_images[:-15] + "resultats" + "/" + name_fig_LAS + "plot_image_" + str(i) + ".pdf", dpi = 1)
    
    # writeVideo(vid,getframe(gcf))
    


# close(vid)


t = [u for u in range (0,np.shape(y)[0])]
x = np.array([u for u in range (0,np.shape(y)[1])]) * mmparpixel
array_final = np.vstack([x, y])
t.insert(0,0)
t = np.array(t) / facq

array_final = np.c_[t, array_final]

        # ENREGISTRE LES DONNEES DANS UN FICHIER .NPY
param_complets.extend([ "haut_LAS = " + str(haut_LAS) ,"bas_LAS = " + str(bas_LAS),"CONVOLUTION : ","size_conv = " + str(size_conv),"precision_conv = " + str(precision_conv),"sigma = " + str(sigma),"TRAITEMENT", "param1 = " + str(param1),"param2 = " + str(param2),"param3 = " + str(param3),"largeur_fit_sub = " + str(a),"border = " + str(border) ])
param_complets = np.array(param_complets)
if save :
    np.save(path_images[:-15] + "resultats" + "/Paramètres.txt", param_complets, "%s")
    np.save(path_images[:-15] + "resultats" + "/positionLAS.npy", array_final)


