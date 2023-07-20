# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:08:41 2022

@author: Turbots
"""




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
    
param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(kernel_size), "kernel_iteration = " + str(kernel_iteration) ,"nbframe = " + str(nbframe) ,"maxsize = " + str(maxsize) , "minsize = " + str(minsize) , "sepwb = " + str(sepwb) , "size_crop = " + str(size_crop) , "bordersize = " + str(bordersize), "mmparpixel = " + str(mmparpixel), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp)]

#%%

display = True
display_result_only = True


gradient = False
maximum = True
red_ratio = False
convconv = True

moyenner = False    #faire un moyennage local (convolve2d)
medfilter = True    #mettre un filtre médian et dilate horizontal

save = False
saveplot = False

size_subplot = 3




#%%

if display and nbframe > 20 :
    continuer = input ("Attention plus de 20 figures à afficher, continuer ?")
    if not (continuer == "oui") :
        exit()

if convconv :
    maximum = True


im = cv2.imread(path_images + liste_images[0])  
im2 = np.array(im)
im_2 = im2 [:,:,2]


y = np.zeros((len(liste_images),(np.shape(im_2)[1])))
xf = np.linspace(-50,50,100)
fconv = (1-xf**2/sigma)*np.exp(-xf**2/(2*sigma))
fconv = fconv - np.mean(fconv)


# plt.figure()
# fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize = set_size(width = 800, fraction = 1, subplots = (2,2)))
# ax1.pcolor,esh()
# ax2.plot(x,y)
# plt.tight_layout()

for i in range (debut, debut + nbframe, 1) : #int(len(liste_images)/ nbframe + 1)) :
    
    #Ouverture image et choix du canal Rouge
    
    im = cv2.imread(path_images + liste_images[i])
    
    # SELECTIONNE LA REGION D'INTERET
    
    im2 = np.array(im)
    im_red = im2[:,:,2]
      
    if display : #display and not display_result_only :
        figurejolie()
        joliplot('', '', (), (), image = im_red, title = "Image rouge")
    
    im_las = np.zeros(((np.shape(im_red)[0]),(np.shape(im_red)[1])))
    im_las_base = np.zeros(((np.shape(im_red)[0]),(np.shape(im_red)[1])))
    
    if moyenner :        
        #Convolution pour smoother l'image  
        siz = 5
        
        im_red = convolve2d (im_red, np.ones((siz,siz))/ siz ** 2 , mode = 'same')
        
    if medfilter :
        #filtre médian et dilate en horizontal
        #autre filtre : detection bord, noise reduction, erodilate
        edges_image = im_red.astype('int32')
        dx = ndimage.sobel(edges_image, 0)
        dy = ndimage.sobel(edges_image, 1)
        edges_image = np.hypot(dx, dy)
        # affichage
        if False :
            figurejolie()
            plt.imshow(edges_image, cmap = 'Greys_r')
            plt.show()
        
        dst = cv2.fastNlMeansDenoising(im_red,None,30,30,21)
        
        if False :
            figurejolie()
            joliplot('', '', (), (), image = dst, title = "image denoise" )
        
        image_binaire = dst < 20
        
        image_binaire = (image_binaire * scale).astype(np.uint8)
        
        if False :
            figurejolie()
            joliplot('', '', (), (), image = image_binaire, title = "image binaire" )
        
        
        
        
        # image_binaire = cv2.bitwise_not(image_binaire)
        
        im_red_traitee = erodedilate(image_binaire, 6, 3, False, path_images, "lala")
        
        edges_image = im_red_traitee.astype('int32')
        dx = ndimage.sobel(edges_image, 0)
        dy = ndimage.sobel(edges_image, 1)
        edges_image = np.hypot(dx, dy)
        # affichage
        if False :
            figurejolie()
            plt.imshow(edges_image, cmap = 'Greys_r')
            plt.show()
        
        # im_red_traitee = medfilt2d(im_red, med_size)
        
        if False : #display and not display_result_only:
            figurejolie()
            joliplot('', '', (), (), image = im_red_traitee, title = "Image filtre médian " + str (med_size) )
            
        
        kernel = np.ones((5,k_size), np.uint8)
        k_iteration = 3
        im_red_traitee = cv2.dilate(im_red_traitee, kernel, iterations=k_iteration)
        
        if display and not display_result_only :
            figurejolie()
            joliplot('', '', (), (), image = im_red_traitee, title = "Image dialted size " + str (k_size) + " iterations " + str(k_iteration))
        
        
        im_red_traitee = dst
     
    [nx,ny] = np.shape(im_red_traitee)
    
    
    for k in range (ny): # colonne
    
    # DETECTION MAX GRADIENT
        if convconv : 
            xXxtentacion = np.arange(0,nx)
            im_conv = np.convolve(im_red_traitee[:,k], fconv, mode = 'same' )
            
            if display and (k == 400 or k == 200) and not display_result_only :
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
            border = 50

            indice =  np.argmax(colonne_k[border:-border])
            indice = indice + border
              
            if display:
                im_las_base[indice,k] = 255 
            
            #sub_pixellaire
            
            a=5 #zone pour le fit polynome
            
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
        im[:,:,2] = im[:,:,2]
        im[:,:,1] = im_las_base
        im[:,:,0] = im_las
        if display :
            figurejolie()
            joliplot('', '', (), (), image = im, title = "Image " + str(i) + " / " + str(len(liste_images)))
        
        if save :
            plt.imsave(path_images[:-15] + "resultats" + "/image_superposée_wow.tiff" , im)   

    
    print("Image " + str(i + 1) + " sur " + str(len(liste_images)))
    
    if saveplot :
        plt.savefig(path_images[:-15] + "resultats" + "/" + name_fig_LAS + "plot_image_" + str(i) + ".png", dpi = 2000)
    
    # writeVideo(vid,getframe(gcf))
    


# close(vid)


t = [u for u in range (0,np.shape(y)[0])]
x = np.array([u for u in range (0,np.shape(y)[1])]) * mmparpixely
array_final = np.vstack([x, y])
t.insert(0,0)
t = np.array(t) / facq

array_final = np.c_[t, array_final]

        # ENREGISTRE LES DONNEES DANS UN FICHIER .txt


param_complets = np.array(param_complets)
if save :
    np.save(path_images[:-15] + "resultats" + "/Paramètres.txt", param_complets, "%s")
    np.save(path_images[:-15] + "resultats" + "/positionLAS.npy", array_final)


